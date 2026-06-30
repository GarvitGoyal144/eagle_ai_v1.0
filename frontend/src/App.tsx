import { useEffect, useState } from "react";
import api from "./api/api";
import LiveCamera from "./components/LiveCamera";

type SystemStatus = {
  project: string;
  version: string;
  backend: string;
  database: string;
  status: string;
};

function App() {
  const [status, setStatus] = useState<SystemStatus | null>(null);

  useEffect(() => {
    api
      .get("/system/status")
      .then((response) => {
        console.log(response.data);
        setStatus(response.data);
      })
      .catch((error) => {
        console.error("API Error:", error);
      });
  }, []);

  return (
    <div className="min-h-screen bg-slate-900 text-white p-8">
      <h1 className="text-4xl font-bold mb-8">🦅 Eagle AI</h1>

      <div className="grid grid-cols-2 gap-8">
        <div>
          <h2 className="text-2xl mb-4">Live Camera</h2>

          <LiveCamera />
        </div>

        <div>
          <h2 className="text-2xl mb-4">System Status</h2>

          {status ? (
            <div>
              <p>
                <strong>Project:</strong> {status.project}
              </p>

              <p>
                <strong>Backend:</strong> {status.backend}
              </p>

              <p>
                <strong>Database:</strong> {status.database}
              </p>

              <p>
                <strong>Version:</strong> {status.version}
              </p>

              <p>
                <strong>Status:</strong> {status.status}
              </p>
            </div>
          ) : (
            <p>Connecting...</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;