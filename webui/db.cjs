var sqlite3 = require('sqlite3');
var mkdirp = require('mkdirp');
var crypto = require('crypto');

mkdirp.sync('./database');

var db = new sqlite3.Database('./database/main.db');

db.serialize(function() {
  // create the database schema for the main app
  db.run("CREATE TABLE IF NOT EXISTS users ( \
    id INTEGER PRIMARY KEY, \
    username TEXT UNIQUE, \
    hashed_password BLOB, \
    salt BLOB \
  )");
  
  db.run("CREATE TABLE IF NOT EXISTS models ( \
    id INTEGER PRIMARY KEY, \
    owner_id INTEGER NOT NULL, \
    title TEXT NOT NULL, \
    description TEXT DEFAULT '', \
    date DATE NOT NULL, \
    stars INTEGER NOT NULL, \
    thumb_image_path TEXT NOT NULL, \
    path TEXT NOT NULL \
  )");
  
  // create an initial user (username: alice, password: letmein)
  var salt = crypto.randomBytes(16);
  db.run('INSERT OR IGNORE INTO users (username, hashed_password, salt) VALUES (?, ?, ?)', [
    'alice',
    crypto.pbkdf2Sync('letmein', salt, 310000, 32, 'sha256'),
    salt
  ]);

});

module.exports = db;