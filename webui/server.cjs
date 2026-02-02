const express = require('express');
const path = require('path');
const cookie_parser = require('cookie-parser');
const body_parser = require('body-parser');
const fs = require('fs');
var db = require('./db.cjs');
var session = require('express-session');
var csrf = require('csurf');
var index_router = require('./routes/index.cjs')
var auth_router = require('./routes/auth.cjs')
var delete_router = require('./routes/delete.cjs')
var passport = require('passport');
var profile_router = require('./routes/profile.cjs');
const gravatar = require('./gravatar.cjs');

// pass the session to the connect sqlite3 module
// allowing it to inherit from session.Store
var SQLiteStore = require('connect-sqlite3')(session);

const app = express();
const port = 8080;
const DEV_SPLATS_DIR = "/Users/chenyu/Projects/DOGS/webui/build/demo/public/assets/splats";
const DEPLOY_SPLATS_DIR = "/Users/chenyu/Projects/DOGS/webui/build/demo/public/assets";
const AVATAR_DIR = "/Users/chenyu/Projects/DOGS/webui/build/demo/public/avatars";
const UPLOADS_DIR = path.join(__dirname, 'public/uploads');

// Also make sure the directory exists
if (!fs.existsSync(DEV_SPLATS_DIR)) {
    fs.mkdirSync(DEV_SPLATS_DIR, { recursive: true });
    console.log('Created splats directory:', DEV_SPLATS_DIR);
}

// Create uploads directory if it doesn't exist
if (!fs.existsSync(UPLOADS_DIR)) {
    fs.mkdirSync(UPLOADS_DIR, { recursive: true });
    console.log('Created uploads directory:', UPLOADS_DIR);
}

if (!fs.existsSync(AVATAR_DIR)) {
    fs.mkdirSync(AVATAR_DIR, { recursive: true });
    console.log('Created splats directory:', AVATAR_DIR);
}

// ========== MIDDLEWARE ORDER IS CRITICAL ==========

// 1. Basic middleware
app.use(body_parser.json());
app.use(express.urlencoded({ extended: true }));

// 2. View engine setup
app.set('view engine', 'ejs');
app.set('views', __dirname + '/views');

// 3. Static files
app.use('/static', express.static(path.join(__dirname, 'public')));
app.use('/splats', express.static(DEV_SPLATS_DIR));
app.use('/splats', express.static(DEPLOY_SPLATS_DIR));
app.use('/uploads', express.static(UPLOADS_DIR));
app.use('/avatars', express.static(AVATAR_DIR))

// 4. Session middleware (MUST come before passport)
app.use(session({
  secret: 'keyboard cat',
  resave: false,
  saveUninitialized: false,
  store: new SQLiteStore({ 
    db: 'sessions.db', 
    dir: './database',
    table: 'sessions'
  }),
  cookie: {
    maxAge: 7 * 24 * 60 * 60 * 1000, // 1 week
    httpOnly: true,
    secure: false // Set to true in production with HTTPS
  }
}));

// 5. Cookie parser (after session)
app.use(cookie_parser());

// 6. Initialize Passport
app.use(passport.initialize());
app.use(passport.session()); // This enables persistent login sessions

// 7. Debug middleware to see what's happening
app.use(function(req, res, next) {
    console.log('=== REQUEST DEBUG ===');
    console.log('Method:', req.method);
    console.log('Path:', req.path);
    console.log('URL:', req.originalUrl);
    console.log('Content-Type:', req.headers['content-type']);
    console.log('Is authenticated:', req.isAuthenticated());
    console.log('=====================');
    next();
});

// 8. Create CSRF middleware
const csrfProtection = csrf({ 
  cookie: {
    key: '_csrf',
    httpOnly: true,
    secure: false,
    sameSite: 'lax'
  }
});

// 9. Apply CSRF protection with exception for file uploads
app.use(function(req, res, next) {
  // List of routes to skip CSRF protection
  const skipCsrfRoutes = [
    { path: '/profile/avatar', method: 'POST' },
    { path: '/upload', method: 'POST' }
  ];
  
  // Check if current route should skip CSRF
  const shouldSkip = skipCsrfRoutes.some(route => 
    req.path === route.path && req.method === route.method
  );
  
  if (shouldSkip) {
    console.log(`Skipping CSRF for ${req.method} ${req.path}`);
    return next();
  }
  
  // Apply CSRF to all other routes
  csrfProtection(req, res, next);
});

// 10. Gravatar middleware (after passport so req.user is available)
app.use(function(req, res, next) {
    if (req.user && req.user.email) {
        res.locals.gravatarUrl = gravatar.getGravatarUrl(req.user.email, 150, 'identicon');
        req.user.gravatarUrl = gravatar.getGravatarUrl(req.user.email, 150, 'identicon');
    }
    // Make gravatar utility available in all templates
    res.locals.gravatar = gravatar;
    next();
});

// 11. Messages middleware
app.use(function(req, res, next) {
  var msgs = req.session.messages || [];
  res.locals.messages = msgs;
  res.locals.hasMessages = !! msgs.length;
  req.session.messages = [];
  next();
});

// 12. CSRF token middleware - make it available in templates
// Only set csrfToken for non-skipped routes
app.use(function(req, res, next) {
  // Check if we should generate CSRF token
  const skipCsrfRoutes = [
    { path: '/profile/avatar', method: 'POST' },
    { path: '/upload', method: 'POST' }
  ];
  
  const shouldSkip = skipCsrfRoutes.some(route => 
    req.path === route.path && req.method === route.method
  );
  
  if (!shouldSkip && req.csrfToken) {
    res.locals.csrfToken = req.csrfToken();
  } else {
    res.locals.csrfToken = ''; // Empty for skipped routes
  }
  next();
});

// 13. Make user available in all templates
app.use(function(req, res, next) {
  res.locals.user = req.user;
  next();
});

// 14. Routes
app.use('/', index_router);
app.use('/', auth_router);
app.use('/', profile_router);
app.use('/', delete_router);

// 14. Viewer route (specific route, comes after general routes)
app.get('/viewer:tagId', (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content, Accept, Content-Type, Authorization');
    res.setHeader('Cross-origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cross-origin-Opener-Policy','same-origin');
    
    let model_name = req.params.tagId;
    console.log("Viewer requested model:", model_name);
    
    // Remove leading colon if present
    if (model_name.startsWith(':')) {
        model_name = model_name.substring(1);
    }
    
    // Remove any query parameters
    model_name = model_name.split('?')[0];
    
    console.log("Cleaned model name:", model_name);
    
    // Check if file exists physically
    const filePath = path.join(DEV_SPLATS_DIR, model_name);
    console.log("Looking for file at:", filePath);
    console.log("File exists:", fs.existsSync(filePath));
    
    // Query database for model details including title
    db.get('SELECT * FROM models WHERE path LIKE ? OR path = ?', 
        [`%/${model_name}`, `/splats/${model_name}`], 
        (err, model) => {
            
        if (err) {
            console.error('Database error:', err);
            const data = {
                message: model_name,
                title: model_name.replace(/\.[^/.]+$/, ""),
                filename: model_name,
                fileExists: fs.existsSync(filePath)
            };
            return res.render('viewer', { data });
        }
        
        if (model) {
            console.log("Found model in database:", model.title);
            const data = {
                message: model.title || model_name.replace(/\.[^/.]+$/, ""),
                title: model.title || model_name.replace(/\.[^/.]+$/, ""),
                filename: model_name,
                dbPath: model.path,
                fileExists: fs.existsSync(filePath),
                description: model.description || '',
                uploadDate: model.date || '',
                stars: model.stars || 0
            };
            res.render('viewer', { data });
        } else {
            console.log("Model not found in database, using filename");
            const data = {
                message: model_name.replace(/\.[^/.]+$/, ""),
                title: model_name.replace(/\.[^/.]+$/, ""),
                filename: model_name,
                fileExists: fs.existsSync(filePath)
            };
            res.render('viewer', { data });
        }
    });
});

app.listen(port, () => {
  console.log('Server running on port %d', port);
});
