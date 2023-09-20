import { useState } from "react"

export default function App () {

    const [newItem , setNewItem] = useState("")
    const [todos,setTodos ]= useState([])

    function handleSubmit(e){
        e.preventDefault()
        setTodos({...todos ,id: crypto.randomUUID(), title: newItem , completed: false})
    }
    return(
        <>
        <form className="inputForm" onSubmit={handleSubmit}>
            <div className="inputDiv">
                <label htmlFor="inputLabel">New Item</label>
                <input value type="text" id="item" ></input>
            </div>
            <button className="bttn">Add</button>
        </form>
        <h1 className="header">ToDo List</h1>
        <ul className="list">
<li>
    <label>
        <input type="checkbox"/>item1
    </label>
    <button type="button" class="btn btn-danger">Danger</button>
</li>
        </ul>
         </>
    )
}