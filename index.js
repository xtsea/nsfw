const express = require('express');
const axios = require('axios');
const nsfwjs = require('nsfwjs');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');

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

app.use(express.urlencoded({ extended: true }));
app.use(express.json());

app.get('/', (req, res) => {
  res.redirect('https://akeno.randydev.my.id');
});

const upload = multer();

app.post('/nsfw', upload.single('file'), async (req, res) => {
    try {
        let imageBuffer;

        if (req.file) {
            imageBuffer = req.file.buffer;
        } else if (req.body.url) {
            const response = await axios.get(req.body.url, { responseType: 'arraybuffer' });
            imageBuffer = Buffer.from(response.data);
        } else {
            return res.status(400).json({ message: 'No file or URL provided.' });
        }

        let imageTensor;

        if ((req.file && req.file.mimetype === 'image/gif') || req.body.url.endsWith('.gif')) {
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

app.use((req, res) => {
  res.sendStatus(404);
});

app.listen(PORT, HOST, () => {
  console.log(`Server is running on http://${HOST}:${PORT}`);
});
