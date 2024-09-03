const express = require('express');
const axios = require('axios');
const nsfwjs = require('nsfwjs');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 9740;
const HOST = '0.0.0.0';

let model;

nsfwjs.load("https://raw.githubusercontent.com/infinitered/nsfwjs/master/models/inception_v3/model.json", { size: 299 })
  .then((loadedModel) => {
    model = loadedModel;
    console.log('Model loaded');
  })
  .catch((error) => {
    console.error('Error loading model:', error);
  });

app.get('/', (req, res) => {
  res.redirect('https://akeno.randydev.my.id');
});

app.get('/nsfw', async (req, res) => {
  try {
    const { url } = req.query;
    if (!url) {
      return res.status(400).json({ message: 'Invalid URL.' });
    }

    const response = await axios.get(url, { responseType: 'arraybuffer' });
    const imageBuffer = Buffer.from(response.data);

    let imageTensor;

    if (url.endsWith('.gif')) {
      const jpgBuffer = await sharp(imageBuffer)
        .resize({ width: 299, height: 299 })
        .toFormat('jpeg')
        .toBuffer();

      imageTensor = tf.node.decodeImage(jpgBuffer, 3);
    } else {
      imageTensor = tf.node.decodeImage(imageBuffer, 3);
    }

    const predictions = await model.classify(imageTensor);
    imageTensor.dispose();

    const formattedPredictions = predictions.reduce((acc, { className, probability }) => {
      acc[className] = probability;
      return acc;
    }, {});

    res.json(formattedPredictions);
  } catch (error) {
    console.error('Error processing image:', error);
    
    if (error.response) {
      return res.status(error.response.status).json({ message: 'Error fetching image from URL.', details: error.message });
    } else if (error.code === 'ERR_INVALID_URL') {
      return res.status(400).json({ message: 'Invalid image URL.', details: error.message });
    } else {
      return res.status(500).json({ message: 'Internal server error.', details: error.message });
    }
  }
});

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads');
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const imageUpload = multer({ storage: storage }).single('image');

app.post('/nsfw-image', imageUpload, async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded.' });
    }
    const imagePath = path.join(__dirname, 'uploads', req.file.filename);
    const imageBuffer = fs.readFileSync(imagePath);
    const imageTensor = tf.node.decodeImage(imageBuffer);
    const predictions = await model.classify(imageTensor);
    const formattedPredictions = predictions.reduce((acc, { className, probability }) => {
      acc[className] = probability;
      return acc;
    }, {});
    fs.unlinkSync(imagePath);
    res.json(formattedPredictions);
  } catch (error) {
    console.error('Error processing image:', error);
    res.status(500).json({ message: 'Internal server error.', details: error.message });
  }
});

app.use((req, res) => {
  res.sendStatus(404);
});

app.listen(PORT, HOST, () => {
  console.log(`Server is running on http://${HOST}:${PORT}`);
});
