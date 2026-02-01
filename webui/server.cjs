const express = require('express');
const path = require('path');
const cookie_parser = require('cookie-parser');
const body_parser = require('body-parser');
const fs = require('fs');
var session = require('express-session');
var csrf = require('csurf');
var index_router = require('./routes/index.cjs')
var auth_router = require('./routes/auth.cjs')
var delete_router = require('./routes/delete.cjs')
var passport = require('passport');

// pass the session to the connect sqlite3 module
// allowing it to inherit from session.Store
var SQLiteStore = require('connect-sqlite3')(session);

const app = express();
const port = 8080;
const DEV_SPLATS_DIR = "/Users/chenyu/Projects/DOGS/webui/build/demo/public/assets/splats";
const DEPLOY_SPLATS_DIR = "/Users/chenyu/Projects/DOGS/webui/build/demo/public/assets";
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

app.use(body_parser.json());
// app.use(cors());
app.use(express.urlencoded({ extended: true }));
// app.use(passport.initialize());
// app.use(passport.session());

app.set('view engine', 'ejs');
app.set('views', __dirname + '/views');
// app.engine('html', require('ejs').renderFile);

// Enable using static files and mount to '/static'.
app.use('/static', express.static(path.join(__dirname, 'public')));
app.use('/splats', express.static(DEV_SPLATS_DIR));
app.use('/splats', express.static(DEPLOY_SPLATS_DIR));
app.use('/uploads', express.static(UPLOADS_DIR));

app.use(session({
  secret: 'keyboard cat',
  resave: false, // don't save session if unmodified
  saveUninitialized: false, // don't create session until something stored
  store: new SQLiteStore({ db: 'sessions.db', dir: './database' })
}));
app.use(cookie_parser())
app.use(csrf({ cookie: true }));
app.use(passport.authenticate('session'));
app.use(function(req, res, next) {
  var msgs = req.session.messages || [];
  res.locals.messages = msgs;
  res.locals.hasMessages = !! msgs.length;
  req.session.messages = [];
  next();
});
app.use(function(req, res, next) {
  // res.locals.csrfToken = req.csrfToken();
  var token = req.csrfToken();
  res.cookie('XSRF-TOKEN', token);
  res.locals.csrfToken = token;
  next();
});

app.use('/', index_router);
app.use('/', auth_router);
app.use('/', delete_router);

app.get('/viewer:tagId', (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content, Accept, Content-Type, Authorization');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, PATCH, OPTIONS');
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
    
    const data = {
        message: model_name,
    }

    res.render('viewer', { data });
});

app.listen(port, () => {
  console.log('Server running on port %d', port);
});
