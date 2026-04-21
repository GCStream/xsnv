const fs = require('fs');
const path = require('path');
const { DOWNLOADS_DIR, LABELS_DIR, IMAGE_EXTENSIONS, LABEL_FILENAME } = require('./config');

class DataManager {
  constructor() {
    this.downloadsDir = DOWNLOADS_DIR;
    this.labelsDir = LABELS_DIR;
  }

  getAlbumSummary() {
    const albums = this.getAllAlbums();
    const labeled = albums.filter(a => a.labeled);
    const unlabeled = albums.filter(a => !a.labeled);
    return {
      total: albums.length,
      labeled: labeled.length,
      unlabeled: unlabeled.length
    };
  }

  getAllAlbums() {
    if (!fs.existsSync(this.downloadsDir)) return [];
    const dirs = fs.readdirSync(this.downloadsDir).filter(name => {
      const dirPath = path.join(this.downloadsDir, name);
      return fs.statSync(dirPath).isDirectory();
    });

    return dirs.map(name => this.getAlbumInfo(name)).sort((a, b) => a.name.localeCompare(b.name, 'zh'));
  }

  getAlbumInfo(albumName) {
    const albumPath = path.join(this.downloadsDir, albumName);
    const labelPath = path.join(this.labelsDir, `${albumName}_${LABEL_FILENAME}`);
    const hasLabel = fs.existsSync(labelPath);
    const imageCount = this.getImageCount(albumPath);

    return {
      name: albumName,
      path: albumPath,
      imageCount,
      labeled: hasLabel
    };
  }

  getImageCount(albumPath) {
    if (!fs.existsSync(albumPath)) return 0;
    return fs.readdirSync(albumPath).filter(file => {
      const ext = path.extname(file).toLowerCase();
      return IMAGE_EXTENSIONS.has(ext);
    }).length;
  }

  getLabeledAlbums() {
    return this.getAllAlbums().filter(a => a.labeled);
  }

  getUnlabeledAlbums() {
    return this.getAllAlbums().filter(a => !a.labeled);
  }

  getAllImages(albumName) {
    const albumPath = path.join(this.downloadsDir, albumName);
    if (!fs.existsSync(albumPath)) return [];
    return fs.readdirSync(albumPath).filter(file => {
      const ext = path.extname(file).toLowerCase();
      return IMAGE_EXTENSIONS.has(ext);
    }).map(file => path.join(albumPath, file));
  }

  getRandomSamples(albumName, count) {
    const images = this.getAllImages(albumName);
    const shuffled = images.sort(() => Math.random() - 0.5);
    return shuffled.slice(0, count);
  }

  loadLabels(albumName) {
    const labelPath = path.join(this.labelsDir, `${albumName}_${LABEL_FILENAME}`);
    if (!fs.existsSync(labelPath)) return {};
    const content = fs.readFileSync(labelPath, 'utf8');
    return JSON.parse(content);
  }

  saveLabel(albumName, labelData) {
    const labelPath = path.join(this.labelsDir, `${albumName}_${LABEL_FILENAME}`);
    try {
      const existing = fs.existsSync(labelPath) ? JSON.parse(fs.readFileSync(labelPath, 'utf8')) : {};
      const merged = { ...existing, ...labelData };
      fs.writeFileSync(labelPath, JSON.stringify(merged, null, 2), 'utf8');
      return true;
    } catch (e) {
      console.error('Failed to save label:', e);
      return false;
    }
  }

  saveImageLabel(albumName, imageName, labelData) {
    const labelPath = path.join(this.labelsDir, `${albumName}_${LABEL_FILENAME}`);
    try {
      const existing = fs.existsSync(labelPath) ? JSON.parse(fs.readFileSync(labelPath, 'utf8')) : {};
      if (!existing.images) existing.images = {};
      existing.images[imageName] = { ...existing.images[imageName], ...labelData };
      fs.writeFileSync(labelPath, JSON.stringify(existing, null, 2), 'utf8');
      return true;
    } catch (e) {
      console.error('Failed to save image label:', e);
      return false;
    }
  }
}

module.exports = DataManager;