#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const workerSrc = path.join(__dirname, '../node_modules/pdfjs-dist/build/pdf.worker.min.mjs');
const workerDest = path.join(__dirname, '../public/pdf.worker.min.js');

try {
  if (fs.existsSync(workerSrc)) {
    fs.copyFileSync(workerSrc, workerDest);
    console.log('PDF.js worker file copied successfully');
  } else {
    console.error('PDF.js worker file not found at:', workerSrc);
    process.exit(1);
  }
} catch (error) {
  console.error('Error copying PDF.js worker file:', error);
  process.exit(1);
} 