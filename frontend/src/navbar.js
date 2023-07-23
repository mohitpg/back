import { useState } from 'react';
import Offcanvas from 'react-bootstrap/Offcanvas';
import Container from 'react-bootstrap/Container';
import Navbar from 'react-bootstrap/Navbar';
import Nav from 'react-bootstrap/Nav';
import "./navbar.css"

function Navibar () {
    const [show, setShow] = useState(false);
    const handleClose = () => setShow(false);
    const handleShow = () => setShow(true);
    return (
        <Navbar variant='light'>
        <Container >
          <Nav className="mx-auto">
            <Navbar.Brand href="#home">Home</Navbar.Brand>
            <Nav.Link href="#home" onClick={handleShow}>About</Nav.Link>
          </Nav>
        </Container>
        <Offcanvas show={show} onHide={handleClose} style={{background: 'black'}}>
        <Offcanvas.Header closeButton>
          <Offcanvas.Title className='offTitle'>About this site</Offcanvas.Title>
        </Offcanvas.Header>
        <Offcanvas.Body className='offBody'>
          This project made with flask and react aims to transcribe/caption the image given by the user.
          However it may lead to inaccurate results sometimes as it's trained on a small dataset. For any suggestions-<br/>
          Contact Me: mohitp.gupta1102@gmail.com
        </Offcanvas.Body>
      </Offcanvas>
        </Navbar>
    )
}

export default Navibar;