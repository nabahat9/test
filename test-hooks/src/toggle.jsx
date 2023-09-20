import React, { useState } from "react";

function Toggle() {
  const [theValue, changeValue] = useState("show");
  const [concept, changeConcept] = useState("");

  const click = () => {
    if (theValue === "show") {
      changeValue("hide");
      changeConcept("This is what we want");
    } else {
      changeValue("show");
      changeConcept("");
    }
  };

  return (
    <div>
      <button onClick={click}>{theValue}</button>
      {concept}
    </div>
  );
}

export default Toggle;
