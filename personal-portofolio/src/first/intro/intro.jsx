import React from 'react';
import './intro.css';
import Github from './github.png';
import linkedIn from './linkedin.png';
import Instagram from './instagram.png';
import Vector1 from './Vector1.png';
import Vector2 from './Vector2.png';
import boy from './boy.png';
import thumbup from './thumbup.png';
import crown from './crown.png';
import glassesimoji from './glassesimoji.png';




const Intro = () => {
  return (
    <div className="intro">
       <div className="i-left">
  <div className="i-name">
    <span>Hy ! I'm</span><span>Nabahat Imane</span>
    <span>Frontend developer with a passion for creating visually appealing and 
      user-friendly web experiences. With a solid grasp of HTML, CSS, and JavaScript,
       I specialize in turning design concepts into interactive and responsive websites.</span>
    </div>
        <button className="button i-button">Hire me</button>
        
        <div className="i-icons">
          <a href='https://github.com/nabahat9'>
        <img src={Github} alt="" />
        </a>
        <a href='https://www.instagram.com/imane.mer4/'>
          <img src={Instagram} alt="" />
          </a>
          <a href='https://www.linkedin.com/in/nabahat-imane-bb1b96273/'>
          <img src={linkedIn} alt="" />
          </a>
</div>
        </div>
        <div className="i-right">
          <img src={Vector1}/>
          <img src={Vector2}/>
          <img src={boy}/>
          <div className="floating"> </div>
        </div>
    </div>
  );
};

export default Intro;
