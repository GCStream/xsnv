/**
 * Configuration settings for the Vision-Language Annotator
 */
const path = require('path');
const fs = require('fs');

const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const DOWNLOADS_DIR = path.join(PROJECT_ROOT, 'downloads');
const LABELS_DIR = path.join(PROJECT_ROOT, 'labels');

const ENV_FILE = path.join(PROJECT_ROOT, '.env');
if (fs.existsSync(ENV_FILE)) {
  const envContent = fs.readFileSync(ENV_FILE, 'utf8');
  envContent.split('\n').forEach(line => {
    const match = line.match(/^([^=]+)=(.*)$/);
    if (match) process.env[match[1]] = match[2];
  });
}

const GRID_COLUMNS = 4;
const IMAGE_HEIGHT = 200;
const SAMPLE_COUNT = 8;

const DEFAULT_QUICK_TAGS = [
  '高质量',
  '一般',
  '模糊',
  '重复',
  '姿势好',
  '服装精美',
  '背景杂乱',
  '需要删除'
];

const SCORE_LABELS = {
  1: '极差',
  2: '差',
  3: '较差',
  4: '一般-',
  5: '一般',
  6: '一般+',
  7: '较好',
  8: '好',
  9: '极好'
};

const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']);
const LABEL_FILENAME = 'labels.json';
const PORT = process.env.PORT || 3000;
const SLIDESHOW_MIN_IMAGES = 30;

module.exports = {
  PROJECT_ROOT,
  DOWNLOADS_DIR,
  LABELS_DIR,
  GRID_COLUMNS,
  IMAGE_HEIGHT,
  SAMPLE_COUNT,
  DEFAULT_QUICK_TAGS,
  SCORE_LABELS,
  IMAGE_EXTENSIONS,
  LABEL_FILENAME,
  PORT,
  SLIDESHOW_MIN_IMAGES
};