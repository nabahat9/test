import React, { useReducer } from 'react';

const ShoppingCard =()=> {
  // Define the initial state of the shopping cart
  const initialState = {
    ShoppingCart: [],
    total: 0,
  };

  // Define the reducer function to manage the shopping cart state
  const cartReducer = (state, action) => {
    switch (action.type) {
      case 'addfirst':
        return {
          ...state,
          ShoppingCart: [...state.ShoppingCart, 'product1'],
          total: state.total + 10,
        };
      case 'addsecond':
        return {
          ...state,
          ShoppingCart: [...state.ShoppingCart, 'product2'],
          total: state.total + 20,
        };
      case 'addthird':
        return {
          ...state,
          ShoppingCart: [...state.ShoppingCart, 'product3'],
          total: state.total + 30,
        };
      case 'removefirst':
        return {
          ...state,
          ShoppingCart: state.ShoppingCart.filter(item => item !== 'product1'),
          total: state.total - 10,
        };
      case 'removesecond':
        return {
          ...state,
          ShoppingCart: state.ShoppingCart.filter(item => item !== 'product2'),
          total: state.total - 20,
        };
      case 'removethird':
        return {
          ...state,
          ShoppingCart: state.ShoppingCart.filter(item => item !== 'product3'),
          total: state.total - 30,
        };
      default:
        return state;
    }
  };

  // Initialize the shopping cart state using useReducer
  const [state, dispatch] = useReducer(cartReducer, initialState);

  return (
    <>
      <h1>Shopping Products</h1>
      <p>product1 10$ <button onClick={() => dispatch({ type: 'addfirst' })}>add</button></p>
      <p>product2 20$ <button onClick={() => dispatch({ type: 'addsecond' })}>add</button></p>
      <p>product3 30$ <button onClick={() => dispatch({ type: 'addthird' })}>add</button></p>

      <h1>Shopping Cart</h1>

      <ul>
        {state.ShoppingCart.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
      <p>Total: {state.total}$</p>
    </>
  );
}

export default ShoppingCard;
