const express = require('express');
const axios = require('axios');
const nsfwjs = require('nsfwjs');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');

const app = express();
const upload = multer({ storage: multer.memoryStorage() });
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

// Redirect root path to GitHub
app.get('/', (req, res) => {
  res.redirect('https://akeno.randydev.my.id');
});

// NSFW classification endpoint
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

app.post('/nsfw-image', upload.single('file'), async (req, res) => {
  try {
    console.log('File received:', req.file);
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded.' });
    }
    const imageBuffer = req.file.buffer;
    console.log('Image buffer received:', imageBuffer);
    if (imageBuffer.length === 0) {
      return res.status(400).json({ message: 'Empty file received.' });
    }
    const jpgBuffer = await sharp(imageBuffer)
      .resize({ width: 299, height: 299 })
      .toFormat('jpeg')
      .toBuffer();
    console.log('Image resized and converted to JPEG');
    const imageTensor = tf.node.decodeImage(jpgBuffer);
    console.log('Image tensor created');
    const predictions = await model.classify(imageTensor);
    console.log('Predictions:', predictions);
    const formattedPredictions = predictions.reduce((acc, { className, probability }) => {
      acc[className] = probability;
      return acc;
    }, {});
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
