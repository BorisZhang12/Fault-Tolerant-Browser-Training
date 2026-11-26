// Chaos harness: launch the coordinator, open N worker tabs (one browser context
// each), and every 10 s close each live tab with probability p/6 so about p of the
// pool dies per minute. A replacement opens 5 s later. A fraction of tabs can open
// with ?slow=<ms>. Chromium runs with --enable-unsafe-webgpu; record the backend.
// TODO(milestone 5): implement. Config is read from experiments/configs/<E>.json.
console.log("chaos harness: not implemented yet (milestone 5)", process.argv.slice(2));
