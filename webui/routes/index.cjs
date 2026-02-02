var express = require('express');
var ensureLogIn = require('connect-ensure-login').ensureLoggedIn;
var db = require('../db.cjs');
var multer = require('multer');
var path = require('path');
var fs = require('fs');
var gravatar = require('../gravatar.cjs');
var router = express.Router();

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        // Determine destination based on file type
        const ext = path.extname(file.originalname).toLowerCase();
        
        // Use absolute paths
        if (['.splat', '.ksplat', '.ply'].includes(ext)) {
            // Use your existing splats directory
            cb(null, '/Users/chenyu/Projects/DOGS/webui/build/demo/public/assets/splats');
        } else if (['.png', '.jpg', '.jpeg', '.gif', '.webp'].includes(ext)) {
            // Use the uploads directory relative to server.cjs
            const uploadsDir = path.join(__dirname, '../public/uploads');
            // Create directory if it doesn't exist
            if (!fs.existsSync(uploadsDir)) {
                fs.mkdirSync(uploadsDir, { recursive: true });
            }
            cb(null, uploadsDir);
        } else {
            cb(new Error('Invalid file type'), false);
        }
    },
    filename: function (req, file, cb) {
        // Create unique filename
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        const ext = path.extname(file.originalname);
        const originalName = path.basename(file.originalname, ext);
        // Clean filename (remove spaces, special characters)
        const cleanName = originalName.replace(/[^a-zA-Z0-9]/g, '_');
        cb(null, cleanName + '-' + uniqueSuffix + ext);
    }
});

const upload = multer({ 
    storage: storage,
    limits: { fileSize: 500 * 1024 * 1024 } // 800MB limit
});

// Upload route
router.post('/upload', upload.array('files'), function(req, res) {
    try {
        const { title, description } = req.body;
        const files = req.files;
        
        console.log('Upload request received');
        
        if (!req.user) {
            return res.status(401).json({ error: 'You must be logged in to upload' });
        }
        
        if (!files || files.length === 0) {
            return res.status(400).json({ error: 'No files uploaded' });
        }
        
        // Parse files
        let modelFile = null;
        let imageFile = null;
        
        files.forEach(file => {
            const ext = path.extname(file.originalname).toLowerCase();
            console.log('Processing file:', file.originalname, '->', file.filename);
            
            if (['.splat', '.ksplat', '.ply'].includes(ext)) {
                modelFile = file;
            } else if (['.png', '.jpg', '.jpeg', '.gif', '.webp'].includes(ext)) {
                imageFile = file;
            }
        });

        if (!modelFile) {
            return res.status(400).json({ error: 'No 3D model file found' });
        }

        if (!imageFile) {
            return res.status(400).json({ error: 'No thumbnail image found' });
        }

        // Get current date
        const currentDate = new Date().toISOString().split('T')[0];
        
        // IMPORTANT: Create correct paths for the viewer
        // The viewer expects files at /splats/filename.splat
        const modelFilename = path.basename(modelFile.path);
        const imageFilename = path.basename(imageFile.path);
        
        // Use relative paths that match your static file serving
        const modelPath = '/splats/' + modelFilename;
        const thumbPath = '/uploads/' + imageFilename;
        
        console.log('File paths for viewer:', {
            modelPath: modelPath,
            thumbPath: thumbPath,
            modelFileExists: fs.existsSync(modelFile.path),
            imageFileExists: fs.existsSync(imageFile.path)
        });
        
        // Save to database
        db.run(
            'INSERT INTO models (owner_id, title, description, date, stars, thumb_image_path, path) VALUES (?, ?, ?, ?, ?, ?, ?)',
            [
                req.user.id, 
                title || 'Untitled Model',
                description || '',
                currentDate, 
                0, 
                thumbPath,
                modelPath
            ],
            function(err) {
                if (err) {
                    console.error('Database error:', err);
                    return res.status(500).json({ error: 'Failed to save model to database: ' + err.message });
                }
                
                console.log('Model saved to database, ID:', this.lastID);
                
                // Verify files are accessible
                setTimeout(() => {
                    const modelUrl = 'http://localhost:8080' + modelPath;
                    const thumbUrl = 'http://localhost:8080' + thumbPath;
                    console.log('Test URLs:', { modelUrl, thumbUrl });
                }, 1000);
                
                res.json({ 
                    message: 'Upload successful!',
                    modelId: this.lastID,
                    title: title || 'Untitled Model',
                    modelPath: modelPath,
                    thumbPath: thumbPath
                });
            }
        );

    } catch (error) {
        console.error('Upload error:', error);
        res.status(500).json({ error: 'Upload failed: ' + error.message });
    }
});

// Your existing fetch_models function
function fetch_models(req, res, next) {
  db.all('SELECT * FROM models', function(err, rows) {
    if (err) { return next(err); }
    
    var models = rows.map(function(row) {
      return {
        id: row.id,
        owner_id: row.owner_id,
        title: row.title || 'Untitled Model',
        description: row.description || '',
        date: row.date || new Date().toISOString().split('T')[0],
        stars: row.stars || 0,
        thumb_image_path: row.thumb_image_path,
        path: row.path
      }
    });
    
    // Get pagination parameters
    const page = parseInt(req.query.page) || 1;
    const pageSize = 9;
    const totalModels = models.length;
    const totalPages = Math.ceil(totalModels / pageSize);
    const currentPage = Math.max(1, Math.min(page, totalPages || 1));
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = Math.min(startIndex + pageSize, totalModels);
    
    // Get paginated models
    const paginatedModels = models.slice(startIndex, endIndex);
    
    res.locals.models = paginatedModels;
    res.locals.page = currentPage;
    res.locals.pageSize = pageSize;
    res.locals.totalModels = totalModels;
    res.locals.totalPages = totalPages;
    res.locals.startIndex = startIndex;
    res.locals.endIndex = endIndex;
    
    next();
  });
}

/* GET home page. */
router.get('/', fetch_models, function(req, res, next) {
  res.render('index', { 
    user: req.user,
    gravatar: gravatar, // Pass gravatar to template
    models: res.locals.models,
    page: res.locals.page,
    pageSize: res.locals.pageSize,
    totalModels: res.locals.totalModels,
    totalPages: res.locals.totalPages,
    startIndex: res.locals.startIndex,
    endIndex: res.locals.endIndex
  });
});

module.exports = router;