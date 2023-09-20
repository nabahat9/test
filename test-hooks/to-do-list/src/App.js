import React, { useRef } from 'react';
import './App.css';
import Todoss from './Todoss';

function App() {
  const inputRef = useRef(null);
  const todosRef = useRef(null);

  const addItem = () => {
    todosRef.current.addnewitem(inputRef.current.value);
    inputRef.current.value = ''; // Clear the input after adding an item
  };

  return (
    <>
      <h1>To Do List</h1>
      <input placeholder='Add Item' ref={inputRef}></input>
      <button onClick={addItem}>Add Item</button>
      <Todoss ref={todosRef} />
    </>
  );
}

export default App;
