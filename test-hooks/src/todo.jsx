import React, { useReducer, useEffect } from "react";

function Timer() {
  const reducer = (state, action) => {
    switch (action.type) {
      case "start":
        return { ...state, isRunning: true };
      case "stop":
        return { ...state, isRunning: false };
      case "reset":
        return { ...state, count: 0 };
      case "tick":
        return { ...state, count: state.count + 1 };
      default:
        return state;
    }
  };

  const initialState = { count: 0, isRunning: false };
  const [state, dispatch] = useReducer(reducer, initialState);

  useEffect(() => {
    let timerId;

    if (state.isRunning) {
      timerId = setInterval(() => {
        dispatch({ type: "tick" });
      }, 1000);
    }

    return () => {
      clearInterval(timerId);
    };
  }, [state.isRunning]);

  return (
    <div>
      <button onClick={() => dispatch({ type: "start" })}>Start</button>
      <button onClick={() => dispatch({ type: "stop" })}>Stop</button>
      <button onClick={() => dispatch({ type: "reset" })}>Reset</button>
      <p>Count: {state.count}</p>
    </div>
  );
}

export default Timer;
