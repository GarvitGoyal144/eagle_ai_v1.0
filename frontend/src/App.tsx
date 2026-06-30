// import { useEffect, useState } from "react";
// import api from "./api/api";

// function App() {

//   const [status, setStatus] = useState<any>(null);

//   useEffect(() => {

//     api.get("/system/status")
//       .then((response) => {
//         setStatus(response.data);
//       });

//   }, []);

//   return (

//     <div className="min-h-screen bg-slate-900 flex items-center justify-center">

//       <div className="bg-slate-800 rounded-xl p-8 text-white w-[450px]">

//         <h1 className="text-3xl font-bold">
//           🦅 Eagle AI
//         </h1>

//         <p className="mt-2 text-gray-400">
//           AI Surveillance Assistant
//         </p>

//         <hr className="my-6"/>

//         {
//           status ?

//           <div>

//             <p><b>Backend :</b> {status.backend}</p>

//             <p><b>Database :</b> {status.database}</p>

//             <p><b>Version :</b> {status.version}</p>

//             <p><b>Status :</b> {status.status}</p>

//           </div>

//           :

//           <p>Connecting...</p>

//         }

//       </div>

//     </div>

//   );
// }

// export default App;


import { useEffect, useState } from "react";
import api from "./api/api";

function App() {
  const [status, setStatus] = useState<any>(null);

  useEffect(() => {
    api.get("/system/status")
      .then((response) => {
        console.log("API Response:", response.data);
        setStatus(response.data);
      })
      .catch((error) => {
        console.error(error);
      });
  }, []);

  return (
    <div>
      <h1>Eagle AI</h1>

      {status ? (
        <>
          <p>{status.project}</p>
          <p>{status.backend}</p>
          <p>{status.database}</p>
          <p>{status.status}</p>
        </>
      ) : (
        <p>Connecting...</p>
      )}
    </div>
  );
}

export default App;