// The code follows the guide: https://www.passportjs.org/tutorials/password/verify/

var express = require('express');
var passport = require('passport');
var LocalStrategy = require('passport-local');
var crypto = require('crypto');
var db = require('../db.cjs');
var multer = require('multer');
var sqlite3 = require('sqlite3');
var router = express.Router();
const fs = require('fs');

// Remove this duplicate line - you're importing db above
// var db = new sqlite3.Database('./database/main.db');

/* Configure password authentication strategy.
 *
 * The `LocalStrategy` authenticates users by verifying a username and password.
 * The strategy parses the username and password from the request and calls the
 * `verify` function.
 *
 * The `verify` function queries the database for the user record and verifies
 * the password by hashing the password supplied by the user and comparing it to
 * the hashed password stored in the database.  If the comparison succeeds, the
 * user is authenticated; otherwise, not.
 */
passport.use(new LocalStrategy(function verify(username, password, cb) {
    db.get('SELECT * FROM users WHERE username = ?', [username], function (err, row) {
        if (err) { return cb(err); }
        if (!row) { return cb(null, false, { message: 'Incorrect username or password.' }); }

        crypto.pbkdf2(password, row.salt, 310000, 32, 'sha256', function (err, hashedPassword) {
            if (err) { return cb(err); }
            if (!crypto.timingSafeEqual(row.hashed_password, hashedPassword)) {
                return cb(null, false, { message: 'Incorrect username or password.' });
            }
            return cb(null, row);
        });
    });
}));

/* Configure session management.
 *
 * When a login session is established, information about the user will be
 * stored in the session.  This information is supplied by the `serializeUser`
 * function, which is yielding the user ID and username.
 *
 * As the user interacts with the app, subsequent requests will be authenticated
 * by verifying the session.  The same user information that was serialized at
 * session establishment will be restored when the session is authenticated by
 * the `deserializeUser` function.
 *
 * Since every request to the app needs the user ID and username, in order to
 * fetch todo records and render the user element in the navigation bar, that
 * information is stored in the session.
 */

// Update serializeUser - store only the user ID in session
passport.serializeUser(function (user, cb) {
    process.nextTick(function () {
        if (!user || !user.id) {
            console.error('Serialize error: Invalid user object', user);
            return cb(new Error('Invalid user object'));
        }
        cb(null, user.id); // Store only ID to keep session small
    });
});

// Update deserializeUser - fetch full user data from database
passport.deserializeUser(function (id, cb) {
    db.get('SELECT * FROM users WHERE id = ?', [id], function (err, user) {
        if (err) { 
            console.error('Deserialize error:', err);
            return cb(err); 
        }
        
        if (!user) {
            console.warn('User not found for id:', id);
            return cb(null, false); // User not found
        }
        
        // Ensure all fields exist (for backward compatibility)
        const completeUser = {
            id: user.id,
            username: user.username,
            email: user.email || null,
            full_name: user.full_name || null,
            bio: user.bio || null,
            avatar_path: user.avatar_path || null,
            created_at: user.created_at || null,
            updated_at: user.updated_at || null
        };
        
        cb(null, completeUser);
    });
});

router.get('/login', function (req, res, next) {
    res.render('login');
});

router.post('/login/password', passport.authenticate('local', {
    successRedirect: '/',
    failureRedirect: '/login',
    failureMessage: true
}));

router.post('/logout', function (req, res, next) {
    console.log("auth.cjs user: %s", req.user);

    req.logout(function (err) {
        if (err) { return next(err); }
        res.redirect('/');
        // req.locals = {};
        // req.user = undefined;
        // res.render('index', {user});
    });
});

/* GET /signup
 *
 * This route prompts the user to sign up.
 *
 * The 'signup' view renders an HTML form, into which the user enters their
 * desired username and password.  When the user submits the form, a request
 * will be sent to the `POST /signup` route.
 */
router.get('/signup', function(req, res, next) {
    res.render('signup');
  });
  
/* POST /signup
 *
 * This route creates a new user account.
 *
 * A desired username and password are submitted to this route via an HTML form,
 * which was rendered by the `GET /signup` route.  The password is hashed and
 * then a new user record is inserted into the database.  If the record is
 * successfully created, the user is logged in.
 */
router.post('/signup', function(req, res, next) {
    // Validate password confirmation
    if (req.body.password !== req.body.confirm_password) {
        req.session.messages = [{type: 'error', text: 'Passwords do not match'}];
        return res.redirect('/signup');
    }
    
    // Check if username already exists
    db.get('SELECT id FROM users WHERE username = ?', [req.body.username], function(err, row) {
        if (err) { return next(err); }
        if (row) {
            req.session.messages = [{type: 'error', text: 'Username already exists'}];
            return res.redirect('/signup');
        }
        
        // Check if email already exists
        if (req.body.email) {
            db.get('SELECT id FROM users WHERE email = ?', [req.body.email], function(err, emailRow) {
                if (err) { return next(err); }
                if (emailRow) {
                    req.session.messages = [{type: 'error', text: 'Email already registered'}];
                    return res.redirect('/signup');
                }
                
                proceedWithSignup();
            });
        } else {
            proceedWithSignup();
        }
    });
    
    function proceedWithSignup() {
        var salt = crypto.randomBytes(16);
        crypto.pbkdf2(req.body.password, salt, 310000, 32, 'sha256', function(err, hashedPassword) {
            if (err) { return next(err); }
            
            // Include additional fields in the INSERT statement
            db.run('INSERT INTO users (username, hashed_password, salt, email, full_name) VALUES (?, ?, ?, ?, ?)', [
                req.body.username,
                hashedPassword,
                salt,
                req.body.email || null,
                req.body.full_name || null
            ], function(err) {
                if (err) { 
                    console.error('Signup error:', err);
                    req.session.messages = [{type: 'error', text: 'Failed to create account'}];
                    return res.redirect('/signup');
                }
                
                // Fetch the complete user record after creation
                db.get('SELECT * FROM users WHERE id = ?', [this.lastID], function(err, user) {
                    if (err) { 
                        console.error('Error fetching user after signup:', err);
                        req.session.messages = [{type: 'error', text: 'Account created but login failed'}];
                        return res.redirect('/login');
                    }
                    
                    req.login(user, function(err) {
                        if (err) { 
                            console.error('Login error after signup:', err);
                            req.session.messages = [{type: 'error', text: 'Account created but login failed'}];
                            return res.redirect('/login');
                        }
                        
                        req.session.messages = [{type: 'success', text: 'Account created successfully!'}];
                        res.redirect('/');
                    });
                });
            });
        });
    }
});

router.get('/upload', function(req, res, next) {
  // res.render('upload', { user: req.user, csrfToken: req.csrfToken() });
  res.render('upload', { user: req.user, upload: false });
});

// Create the 'build/demo/public/assets/splats' directory if it doesn't exist
const uploadDir = 'build/demo/public/assets/splats';
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// Set up Multer storage and define the upload path
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    // The directory must be existed in the current directory.
    cb(null, 'build/demo/public/assets/splats'); // uploads directory
  },
  filename: function(req, file, cb) {
    cb(null, file.originalname);
  }
});

const upload = multer({
  storage: storage ,
  limits: { fileSize: 1024 * 1024 * 1024 }, // 1GB
  // fileFilter: (req, file, cb) => {
  //   if (file.mimetype == "image/png" || file.mimetype == "image/jpg" || file.mimetype == "image/jpeg") {
  //     cb(null, true);
  //   } else {
  //     cb(null, false);
  //     const err = new Error('Only .png, .jpg, .jpeg and .splat files are allowed!');
  //     err.name = "extensionError";
  //     return cb(err);
  //   }
  // },
});

function get_title(filename) {
  const dot_index = filename.lastIndexOf('.');
  const slash_index =  filename.lastIndexOf('/');
  var title = filename.substring(slash_index + 1, dot_index);
  
  return title;
}

function insert_model_into_db(user_id, splat_filename, thumb_image_filename) {
  const title = get_title(splat_filename);
  const current_date = new Date();
  const formated_date = current_date.getFullYear() + "-" 
      + current_date.getMonth() + "-" + current_date.getDay();
  // console.log('user_id: %s', user_id);
  // console.log('title: %s', title);
  // console.log('formatted_date: %s', formated_date);
  // console.log('thumb_image_filename: %s', thumb_image_filename);
  // console.log('splat_filename: %s', splat_filename.substring(0, splat_filename.lastIndexOf('.')));
  new_thumb_image_filename = './splats/' + thumb_image_filename.substring(thumb_image_filename.lastIndexOf('/'));
  new_splat_filename = './splats/' + get_title(splat_filename);
  console.log('new_image_filename: %s', new_thumb_image_filename);
  console.log('new_splat_filename: %s', new_splat_filename);
  db.run('INSERT OR IGNORE INTO models (owner_id, title, date, stars, thumb_image_path, path) VALUES (?, ?, ?, ?, ?, ?)',
  [
    user_id,
    title,
    formated_date,
    0,
    new_thumb_image_filename,
    new_splat_filename,
  ]);
}

// Define a route for file upload.
router.post('/upload', upload.array('files', 2), (req, res) => {
  console.log("auth.cjs user: %s", req.user);
  console.log("req.file: %s", req.files);
  if (req.files) {
    console.log(req.files);
  }

  const filename0 = req.files[0].path;
  const filename1 = req.files[1].path;
  const postfix0 = filename0.substring(filename0.lastIndexOf('.'));
  const postfix1 = filename1.substring(filename0.lastIndexOf('.'));
  var valid_files = false;
  if (
    (postfix0 == '.splat' && (postfix1 == '.png' || postfix1 == '.jpg' || postfix1 == '.jpeg')) ||
    ((postfix0 == '.png' || postfix0 == '.jpg' || postfix0 == '.jpeg') && postfix1 == '.splat')) {
      valid_files = true;
  }

  var splat_filename = '';
  var image_filename = '';
  if (filename0.substring(filename0.lastIndexOf('.')) == '.splat') {
    splat_filename = filename0;
    image_filename = filename1;
  } else {
    splat_filename = filename1;
    image_filename = filename0;
  }
  console.log(filename0.substring(filename0.lastIndexOf('.')));
  console.log('splat_filename: %s', splat_filename);
  console.log('image_filename: %s', image_filename)
  insert_model_into_db(req.user.id, splat_filename, image_filename);

  // return res.render('upload', { user: req.user, upload: true });
  // return res.redirect('/upload');
  res.json({ message: 'File uploaded successfully!' });
});

module.exports = router;
