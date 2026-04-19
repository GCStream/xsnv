/**
 * Express Server - Vision-Language Annotator API
 */
const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const DataManager = require('./core/data-manager');
const {
  PORT,
  DOWNLOADS_DIR,
  LABELS_DIR,
  SCORE_LABELS,
  DEFAULT_QUICK_TAGS,
  SLIDESHOW_MIN_IMAGES
} = require('./core/config');

const app = express();
const dataManager = new DataManager();

const isLocalhost = (req) => {
  const ip = req.ip || req.connection.remoteAddress;
  return ip === '127.0.0.1' || ip === '::1' || ip === '::ffff:127.0.0.1';
};

const requireAuth = (req, res, next) => {
  const authToken = process.env.AUTH_TOKEN;
  if (!authToken) return next();
  if (isLocalhost(req)) return next();

  const token = req.headers.authorization?.replace('Bearer ', '');
  if (token === authToken) return next();

  res.status(401).json({ error: 'Unauthorized' });
};

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/config', requireAuth, (req, res) => {
  res.json({
    scoreLabels: SCORE_LABELS,
    quickTags: DEFAULT_QUICK_TAGS,
    gridColumns: 4,
    sampleCount: 8,
    slideshowMinImages: SLIDESHOW_MIN_IMAGES
  });
});

app.get('/api/summary', requireAuth, (req, res) => {
  const summary = dataManager.getAlbumSummary();
  res.json(summary);
});

app.get('/api/albums', requireAuth, (req, res) => {
  const albums = dataManager.getAllAlbums();
  res.json(albums);
});

app.get('/api/albums/unlabeled', requireAuth, (req, res) => {
  const albums = dataManager.getUnlabeledAlbums();
  res.json(albums);
});

app.get('/api/albums/labeled', requireAuth, (req, res) => {
  const albums = dataManager.getLabeledAlbums();
  res.json(albums);
});

app.get('/api/albums/:name', requireAuth, (req, res) => {
  const albumName = decodeURIComponent(req.params.name);
  const albums = dataManager.getAllAlbums();
  const album = albums.find(a => a.name === albumName);

  if (!album) {
    return res.status(404).json({ error: 'Album not found' });
  }

  res.json(album);
});

app.get('/api/albums/:name/samples', requireAuth, (req, res) => {
  const albumName = decodeURIComponent(req.params.name);
  const count = parseInt(req.query.count) || 8;
  const samples = dataManager.getRandomSamples(albumName, count);

  const relativePaths = samples.map(p => path.relative(DOWNLOADS_DIR, p));
  res.json(relativePaths);
});

app.get('/api/albums/:name/images', requireAuth, (req, res) => {
  const albumName = decodeURIComponent(req.params.name);
  const images = dataManager.getAllImages(albumName);

  const relativePaths = images.map(p => path.relative(DOWNLOADS_DIR, p));
  res.json(relativePaths);
});

app.get('/api/albums/:name/labels', requireAuth, (req, res) => {
  const albumName = decodeURIComponent(req.params.name);
  const labels = dataManager.loadLabels(albumName);
  res.json(labels);
});

app.post('/api/albums/:name/labels', requireAuth, (req, res) => {
  const albumName = decodeURIComponent(req.params.name);
  const { score, tags, notes, images } = req.body;

  const labelData = {};
  if (score !== undefined) labelData.albumScore = score;
  if (tags) labelData.albumTags = tags;
  if (notes) labelData.albumNotes = notes;
  if (images) labelData.images = images;

  const success = dataManager.saveLabel(albumName, labelData);

  if (success) {
    res.json({ success: true });
  } else {
    res.status(500).json({ error: 'Failed to save labels' });
  }
});

app.post('/api/albums/:name/images/:image/labels', requireAuth, (req, res) => {
  const albumName = decodeURIComponent(req.params.name);
  const imageName = decodeURIComponent(req.params.image);
  const { score, tags, notes } = req.body;

  const success = dataManager.saveImageLabel(albumName, imageName, { score, tags, notes });

  if (success) {
    res.json({ success: true });
  } else {
    res.status(500).json({ error: 'Failed to save image label' });
  }
});

app.get('/images/*', requireAuth, (req, res) => {
  const relativePath = req.params[0];
  const imagePath = path.join(DOWNLOADS_DIR, relativePath);

  if (!imagePath.startsWith(DOWNLOADS_DIR)) {
    return res.status(403).json({ error: 'Access denied' });
  }

  if (!fs.existsSync(imagePath)) {
    return res.status(404).json({ error: 'Image not found' });
  }

  const ext = path.extname(imagePath).toLowerCase();
  const contentTypes = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.webp': 'image/webp',
    '.gif': 'image/gif',
    '.bmp': 'image/bmp'
  };

  res.set('Content-Type', contentTypes[ext] || 'image/jpeg');
  res.sendFile(imagePath);
});

app.listen(PORT, () => {
  console.log(`Vision-Language Annotator running at http://localhost:${PORT}`);
});