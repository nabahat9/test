import React, { useState } from "react";

function InputUpdate() {
  const [input, changeInput] = useState("");

  const change = (event) => {
    changeInput(event.target.value);
  };

  return (
    <div>
      <input onChange={change} placeholder="write here" />
      {input}
    </div>
  );
}

export default InputUpdate;
