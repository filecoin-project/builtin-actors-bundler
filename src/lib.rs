use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use async_std::channel::bounded;
use async_std::task;
use async_std::task::block_on;

use anyhow::{anyhow, Context, Result};
use cid::Cid;
use fvm_ipld_blockstore::{Block, Blockstore, MemoryBlockstore};
use fvm_ipld_car::CarHeader;
use fvm_ipld_encoding::tuple::*;
use fvm_ipld_encoding::DAG_CBOR;
use multihash_codetable::Code;

const IPLD_RAW: u64 = 0x55;

/// A library to bundle the Wasm bytecode of builtin actors into a CAR file.
///
/// The single root CID of the CAR file points to an CBOR-encoded IPLD
/// Vec<(String, Cid)> where.
pub struct Bundler {
    /// Staging blockstore.
    blockstore: MemoryBlockstore,
    /// Tracks the mapping of actors to Cids. Inverted when writing. Allows overriding.
    added: BTreeMap<u32, (String, Cid)>,
    /// Path of the output bundle.
    bundle_dst: PathBuf,
}

impl Bundler {
    pub fn new<P>(bundle_dst: P) -> Bundler
    where
        P: AsRef<Path>,
    {
        Bundler {
            bundle_dst: bundle_dst.as_ref().to_owned(),
            blockstore: Default::default(),
            added: Default::default(),
        }
    }

    /// Adds bytecode from a byte slice.
    pub fn add_from_bytes(
        &mut self,
        actor_type: u32,
        actor_name: String,
        forced_cid: Option<&Cid>,
        bytecode: &[u8],
    ) -> Result<Cid> {
        let cid = match forced_cid {
            Some(cid) => self.blockstore.put_keyed(cid, bytecode).and(Ok(*cid)),
            None => {
                self.blockstore.put(Code::Blake2b256, &Block { codec: IPLD_RAW, data: bytecode })
            }
        }
        .with_context(|| {
            format!("failed to put bytecode for actor {:?} into blockstore", actor_type)
        })?;
        self.added.insert(actor_type, (actor_name, cid));
        Ok(cid)
    }

    /// Adds bytecode from a file.
    pub fn add_from_file<P: AsRef<Path>>(
        &mut self,
        actor_type: u32,
        actor_name: String,
        forced_cid: Option<&Cid>,
        bytecode_path: P,
    ) -> Result<Cid> {
        let bytecode = std::fs::read(bytecode_path).context("failed to open bytecode file")?;
        self.add_from_bytes(actor_type, actor_name, forced_cid, bytecode.as_slice())
    }

    /// Commits the added bytecode entries and writes the CAR file to disk.
    pub fn finish(self) -> Result<()> {
        block_on(self.write_car())
    }

    async fn write_car(self) -> Result<()> {
        if let Some((actual, expected)) = self.added.keys().copied().zip(1..).find(|(a, b)| a != b)
        {
            return Err(anyhow!(
                "actor types are not sequential: expected {expected}, got {actual}"
            ));
        }

        let mut out = async_std::fs::File::create(&self.bundle_dst).await?;

        let manifest_payload: Vec<&(String, Cid)> = self.added.values().collect();
        let manifest_data = serde_ipld_dagcbor::to_vec(&manifest_payload)?;
        let manifest_link = self
            .blockstore
            .put(Code::Blake2b256, &Block { codec: DAG_CBOR, data: &manifest_data })?;

        let manifest = Manifest { version: 1, data: manifest_link };
        let manifest_bytes = serde_ipld_dagcbor::to_vec(&manifest)?;

        let root = self
            .blockstore
            .put(Code::Blake2b256, &Block { codec: DAG_CBOR, data: &manifest_bytes })?;

        // Create a CAR header.
        let car = CarHeader { roots: vec![root], version: 1 };

        let (tx, mut rx) = bounded(16);
        let write_task =
            task::spawn(async move { car.write_stream_async(&mut out, &mut rx).await.unwrap() });

        // Add the root payload.
        tx.send((root, manifest_bytes)).await.unwrap();

        // Add the manifest payload.
        tx.send((manifest_link, manifest_data)).await.unwrap();

        // Add the bytecodes.
        for cid in self.added.values().map(|(_, cid)| cid) {
            let data = self.blockstore.get(cid).unwrap().unwrap();
            tx.send((*cid, data)).await.unwrap();
        }

        drop(tx);

        write_task.await;

        Ok(())
    }
}

/// The Manifest struct is the versioned envelope for builtin actor manifests.
/// The only currently supported version is 1, which encodes the data as a tuple
/// of strings (actor names) and actor code cids.
#[derive(Serialize_tuple, Deserialize_tuple, Clone)]
struct Manifest {
    pub version: u32,
    pub data: Cid,
}

#[test]
fn test_bundler() {
    use async_std::fs::File;
    use cid::multihash::Multihash;
    use fvm_ipld_car::{load_car_unchecked, CarReader};
    use rand::Rng;

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("test_bundle.car");

    // Write 10 random payloads to the bundle.
    let mut cids = Vec::with_capacity(10);
    let mut bundler = Bundler::new(&path);

    // First 5 have real CIDs, last 5 have forced CIDs.
    for i in 0..10 {
        let forced_cid = (i > 5).then(|| {
            // identity hash
            Cid::new_v1(IPLD_RAW, Multihash::wrap(0, format!("actor-{}", i).as_bytes()).unwrap())
        });
        let cid = bundler
            .add_from_bytes(
                i + 1,
                format!("actor-{i}"),
                forced_cid.as_ref(),
                &rand::thread_rng().gen::<[u8; 32]>(),
            )
            .unwrap();

        dbg!(cid.to_string());
        cids.push(cid);
    }
    bundler.finish().unwrap();

    // Read with the CarReader directly and verify there's a single root.
    let reader = block_on(async {
        let file = File::open(&path).await.unwrap();
        CarReader::new(file).await.unwrap()
    });
    assert_eq!(reader.header.roots.len(), 1);
    dbg!(reader.header.roots[0].to_string());

    // Load the CAR into a Blockstore.
    let bs = MemoryBlockstore::default();
    let roots = block_on(async {
        let file = File::open(&path).await.unwrap();
        load_car_unchecked(&bs, file).await.unwrap()
    });
    assert_eq!(roots.len(), 1);

    // Compare that the previous root matches this one.
    assert_eq!(reader.header.roots[0], roots[0]);

    // The single root represents the manifest.
    let manifest_cid = roots[0];
    let manifest_bytes = bs.get(&manifest_cid).unwrap().unwrap();

    // Deserialize the manifest.
    let manifest: Manifest = serde_ipld_dagcbor::from_slice(manifest_bytes.as_slice()).unwrap();
    assert_eq!(manifest.version, 1);

    let manifest_data = bs.get(&manifest.data).unwrap().unwrap();
    let manifest_vec: Vec<(String, Cid)> =
        serde_ipld_dagcbor::from_slice(manifest_data.as_slice()).unwrap();
    for (i, (target_cid, (name, cid))) in cids.into_iter().zip(manifest_vec.into_iter()).enumerate()
    {
        assert_eq!(format!("actor-{i}"), name);
        assert_eq!(target_cid, cid);
        if i > 5 {
            let expected = Cid::new_v1(
                IPLD_RAW,
                Multihash::wrap(0, format!("actor-{i}").as_bytes()).expect("name too long"),
            );
            assert_eq!(cid, expected)
        }
    }
}
