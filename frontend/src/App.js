import { useEffect, useState } from "react";
import axios from 'axios';
import Navibar from './navbar.js';
import Button from 'react-bootstrap/Button';
import Card from 'react-bootstrap/Card';
import Form from 'react-bootstrap/Form';
import ProgressBar from 'react-bootstrap/ProgressBar';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import uploadimg from "./upload.jpg";

function App() {
  const [file, setFile] = useState(uploadimg);
  const [loadingp, setLoading] = useState(false);
  const [loading2, setLoading2] = useState(false);
  const [loading3, setLoading3] = useState("");

  const submitData = async (e) => {
    e.preventDefault();
    let blob = await fetch(file).then(r => r.blob());
    const data=blob;
    console.log(data);

    const uploadImage = async (imageData) => {
      try {
        setLoading(true);
        setLoading2(false);
        const response = await axios.post('/api', imageData,{
              headers: {
                'Content-Type':'application/json',
              }
            });
        setLoading3(response.data.processed_data.charAt(0).toUpperCase()+response.data.processed_data.slice(1));
        setLoading2(true);
        if(file==uploadimg){
          setLoading3("Please select an image!");
          setLoading2(true);
        }
      } catch (error) {
        console.error(error);
      }
    };

    const reader = new FileReader();
    reader.onloadend = () => {
    uploadImage(reader.result);
    };
    reader.readAsDataURL(data);
  
   };


  function handleChange(e) {
      console.log(e.target.files.length);
      (e.target.files.length==1)?setFile(URL.createObjectURL(e.target.files[0])):setFile(uploadimg);
   }


  return (
    <div className="App" >
      <Navibar />
      <Card text='light' className="maincard border-0">
        <Card.Img variant="top" src={file} style={{width:'25%', height:'25%', margin:'3% auto 2% auto'}}/>
        <Form >
          <input type="file" name="imageinput" onChange={handleChange} style={{margin:'1px -10% 1px auto'}}/>
        </Form>
        <Card.Body>
          <Card.Title style={{fontFamily: 'Oswald', fontSize: '3rem'}}>Image Captioner</Card.Title>
          <Card.Text style={{fontFamily: 'Alegreya Sans', fontSize: '1.5rem'}}>
          Upload an image and generate captions for it!
          </Card.Text>
          <Button size='lg' variant="outline-light" onClick={submitData} style={{margin: "5px auto 3% auto", fontFamily: 'Oswald', fontSize: '1rem'}}>Predict</Button>
          {loadingp?loading2?<p style={{fontFamily: 'Roboto Condensed', fontSize: '2rem'}}>{loading3}</p>:<ProgressBar variant="dark" animated now={100} />:<div></div>}
        </Card.Body>
      </Card>
    </div>
  );
}

export default App;
