import React, { useState } from 'react';
import Toggle from './toggle.jsx';
import InputUpdate from './input.jsx';
import Timer from './todo.jsx';
import Survey from './survey.jsx';
import ShoppingCard from './shoppingCard.jsx';


function App() {
  const [theValue, modify] = useState(0);

  const increment = () => {
    modify(theValue + 1);
  };

  const decrement = () => {
    modify(theValue - 1);
  };

  return (
    <div>
      <button onClick={increment}>Increment</button>
      {theValue}
      
      <button onClick={decrement}>Decrement</button>
   <Toggle/>
   <InputUpdate/>
  <Timer/>
  <Survey/>
  <ShoppingCard/>

    </div>
  );
}

export default App;
