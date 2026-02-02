var sqlite3 = require('sqlite3');
var mkdirp = require('mkdirp');
var crypto = require('crypto');

mkdirp.sync('./database');

var db = new sqlite3.Database('./database/main.db');

db.serialize(function() {
  db.run("CREATE TABLE IF NOT EXISTS users ( \
    id INTEGER PRIMARY KEY, \
    username TEXT UNIQUE, \
    hashed_password BLOB, \
    salt BLOB, \
    email TEXT, \
    full_name TEXT, \
    bio TEXT, \
    avatar_path TEXT DEFAULT NULL, \
    use_gravatar BOOLEAN DEFAULT 0, \
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, \
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP \
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
  
  // Add triggers for updated_at
  db.run(`
    CREATE TRIGGER IF NOT EXISTS update_users_timestamp 
    AFTER UPDATE ON users 
    BEGIN
      UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;
  `);
  
  // create an initial user (username: alice, password: letmein)
  var salt = crypto.randomBytes(16);
  db.run('INSERT OR IGNORE INTO users (username, hashed_password, salt, email, full_name) VALUES (?, ?, ?, ?, ?)', [
    'alice',
    crypto.pbkdf2Sync('letmein', salt, 310000, 32, 'sha256'),
    salt,
    'alice@example.com',
    'Alice Smith'
  ]);
});

module.exports = db;