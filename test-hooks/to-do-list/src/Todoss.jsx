import React, { forwardRef, useImperativeHandle, useState, useRef } from "react";

const Todoss = forwardRef((props, ref) => {
  const [todos, setTodo] = useState([]);

  useImperativeHandle(ref, () => {
    const addnewitem = (newItem) => {
      setTodo([...todos, newItem]);
    };
    return {
      addnewitem
    };
  });

  return (
    <div>
      {/* Render your todos */}
      {todos.map((todo, index) => (
        <div key={index}>{todo}</div>
      ))}
    </div>
  );
});

export default Todoss;
