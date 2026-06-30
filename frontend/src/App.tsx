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
import LiveCamera from "./components/LiveCamera";

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

<div className="min-h-screen bg-slate-900 text-white p-8">

<h1 className="text-4xl font-bold mb-8">
🦅 Eagle AI
</h1>

<div className="grid grid-cols-2 gap-8">

<div>

<h2 className="text-2xl mb-4">
Live Camera
</h2>

<LiveCamera/>

</div>

<div>

<h2 className="text-2xl mb-4">
System Status
</h2>

{status && (

<div>

<p>Backend : {status.backend}</p>

<p>Database : {status.database}</p>

<p>Version : {status.version}</p>

<p>Status : {status.status}</p>

</div>

)}

</div>

</div>

</div>

);
}

export default App;