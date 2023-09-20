import "./topbar.css"
import {search} from "@material-ui/icons";

export default function Topbar(){
    return(
        <div>
            <div className="topbarLeft">
<span className="logo">IMANyySocial</span>
            </div>
            <div className="topbarCenter">
                <div className="searchbar">
                    <search/>
                    <input placeholder="search for friend,posts or video" className="searchInput"/>
                </div>

            </div>
            <div className="topbarRight">
                <div className="topbarLinks">
                    <span className="topbarLinks">HomePage</span>
                    <span className="topbarLinks">TimeLine</span>
                </div>
            </div>
            
        </div>
    )
}