import React, { useReducer } from "react";

function Survey() {
  const reducer = (state, action) => {
    switch (action.type) {
      case "companynotfilled":
        return { ...state, message: "Fill company" };
      case "namenotfilled":
        return { ...state, message: state.message + "Fill name" };
      case "emailnotfilled":
        return { ...state, message: state.message + "Fill email" };
      case "emailnotcorrect":
        return { ...state, message: state.message + "Correct email" };
      default:
        return state;
    }
  };

  const initialState = {
    isCompanyFilled: false,
    isNameFilled: false,
    isEmailFilled: false,
    isEmailTrue: true,
    message: ""
  };

  const [state, dispatch] = useReducer(reducer, initialState);

  const handleInputChange1 = (event) => {
    if (event.target.value === "") {
      dispatch({ type: "companynotfilled" });
    } else {
      dispatch({ type: "companyfilled" });
    }
  };

  const handleInputChange2 = (event) => {
    if (event.target.value === "") {
      dispatch({ type: "namenotfilled" });
    } 
  };

  const handleInputChange3 = (event) => {
    if (event.target.value === "") {
      dispatch({ type: "emailnotfilled" });
    } else {
      dispatch({ type: "emailfilled" });
    }
  };

  return (
    <>
      <h1>Company</h1>
      <input type="text" onChange={handleInputChange1} />
      <h1>Name</h1>
      <input type="text" onChange={handleInputChange2} />
      <h1>Email</h1>
      <input type="email" onChange={handleInputChange3} />
      <p>{state.message}</p>
      
    </>
  );
}

export default Survey;
