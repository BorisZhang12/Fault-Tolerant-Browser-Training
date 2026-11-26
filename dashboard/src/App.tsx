// One page: worker grid on the left (green = training, grey = idle, red = dead),
// live test accuracy on the right with a red tick for each worker death.
// TODO(milestone 7): subscribe to /ws/dashboard and render the grid, chart and counters.
export function App() {
  return (
    <main style={{ fontFamily: "system-ui", padding: 24 }}>
      <h1>Fault-tolerant browser training</h1>
      <p>Dashboard coming in milestone 7.</p>
    </main>
  );
}
