var express = require('express');
var ensureLogIn = require('connect-ensure-login').ensureLoggedIn;
var db = require('../db.cjs');
var multer = require('multer');
var path = require('path');
var fs = require('fs');
var gravatar = require('../gravatar.cjs');

var router = express.Router();

// Create avatars directory if it doesn't exist
const avatarsDir = path.join(__dirname, '../public/avatars');
if (!fs.existsSync(avatarsDir)) {
    fs.mkdirSync(avatarsDir, { recursive: true });
    console.log('Created avatars directory:', avatarsDir);
}

// Configure multer - SIMPLE CONFIGURATION
const upload = multer({
    dest: avatarsDir, // Use temp directory first
    limits: {
        fileSize: 5 * 1024 * 1024, // 5MB
    },
    fileFilter: function(req, file, cb) {
        // Accept images only
        if (!file.originalname.match(/\.(jpg|jpeg|png|gif|webp)$/i)) {
            return cb(new Error('Only image files are allowed!'), false);
        }
        cb(null, true);
    }
});

// Profile page
router.get('/profile', ensureLogIn('/login'), function(req, res, next) {
    db.get('SELECT * FROM users WHERE id = ?', [req.user.id], function(err, user) {
        if (err) {
            console.error('Profile DB error:', err);
            return next(err);
        }
        
        if (!user) {
            console.error('User not found for profile:', req.user.id);
            req.session.messages = [{type: 'error', text: 'User not found'}];
            return res.redirect('/');
        }
        
        // Get user's models count
        db.get('SELECT COUNT(*) as count FROM models WHERE owner_id = ?', [req.user.id], function(err, result) {
            if (err) {
                console.error('Models count error:', err);
                return next(err);
            }
            
            res.render('profile', {
                user: user,
                gravatar: gravatar,
                modelsCount: result.count,
                csrfToken: req.csrfToken ? req.csrfToken() : ''
            });
        });
    });
});

// Update profile information
router.post('/profile/update', ensureLogIn('/login'), function(req, res, next) {
    const { full_name, email, bio } = req.body;
    
    // Validate email if provided
    if (email && !email.includes('@')) {
        req.session.messages = [{type: 'error', text: 'Invalid email address'}];
        return res.redirect('/profile');
    }
    
    db.run(
        'UPDATE users SET full_name = ?, email = ?, bio = ? WHERE id = ?',
        [full_name || '', email || '', bio || '', req.user.id],
        function(err) {
            if (err) {
                console.error('Profile update DB error:', err);
                req.session.messages = [{type: 'error', text: 'Failed to update profile'}];
                return res.redirect('/profile');
            }
            
            req.session.messages = [{type: 'success', text: 'Profile updated successfully'}];
            res.redirect('/profile');
        }
    );
});

// UPLOAD AVATAR - SIMPLIFIED VERSION (no CSRF check for now)
router.post('/profile/avatar', 
    ensureLogIn('/login'),
    upload.single('avatar'),
    function(req, res, next) {
        console.log('=== AVATAR UPLOAD PROCESSING ===');
        console.log('User:', req.user.username);
        console.log('File:', req.file);
        
        if (!req.file) {
            req.session.messages = [{type: 'error', text: 'No file selected'}];
            return res.redirect('/profile');
        }
        
        // Validate file type
        const allowedExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp'];
        const ext = path.extname(req.file.originalname).toLowerCase();
        if (!allowedExtensions.includes(ext)) {
            // Clean up temp file
            fs.unlinkSync(req.file.path);
            req.session.messages = [{type: 'error', text: 'Invalid file type. Only JPG, PNG, GIF, and WebP are allowed.'}];
            return res.redirect('/profile');
        }
        
        // Create final filename
        const finalFilename = 'avatar-' + req.user.id + '-' + Date.now() + ext;
        const finalPath = path.join(avatarsDir, finalFilename);
        const avatarUrl = '/avatars/' + finalFilename;
        
        console.log('Final path:', finalPath);
        console.log('Avatar URL:', avatarUrl);
        
        // Move temp file to final location
        fs.rename(req.file.path, finalPath, function(err) {
            if (err) {
                console.error('Error moving file:', err);
                // Clean up temp file
                if (fs.existsSync(req.file.path)) {
                    fs.unlinkSync(req.file.path);
                }
                req.session.messages = [{type: 'error', text: 'Failed to save image'}];
                return res.redirect('/profile');
            }
            
            // Delete old avatar if exists
            if (req.user.avatar_path && req.user.avatar_path.startsWith('/avatars/')) {
                const oldAvatarPath = path.join(__dirname, '../public', req.user.avatar_path);
                if (fs.existsSync(oldAvatarPath)) {
                    fs.unlink(oldAvatarPath, (err) => {
                        if (err) console.error('Error deleting old avatar:', err);
                    });
                }
            }
            
            // Update database
            db.run(
                'UPDATE users SET avatar_path = ? WHERE id = ?',
                [avatarUrl, req.user.id],
                function(err) {
                    if (err) {
                        console.error('Database update error:', err);
                        // Clean up the new file
                        if (fs.existsSync(finalPath)) {
                            fs.unlinkSync(finalPath);
                        }
                        req.session.messages = [{type: 'error', text: 'Failed to update avatar in database'}];
                        return res.redirect('/profile');
                    }
                    
                    console.log('Avatar updated successfully');
                    req.session.messages = [{type: 'success', text: 'Avatar updated successfully!'}];
                    res.redirect('/profile');
                }
            );
        });
    }
);

// Delete avatar
router.post('/profile/avatar/delete', ensureLogIn('/login'), function(req, res, next) {
    // Delete current avatar file if exists
    if (req.user.avatar_path && req.user.avatar_path.startsWith('/avatars/')) {
        const oldAvatarPath = path.join(__dirname, '../public', req.user.avatar_path);
        if (fs.existsSync(oldAvatarPath)) {
            fs.unlink(oldAvatarPath, (err) => {
                if (err) console.error('Error deleting avatar:', err);
            });
        }
    }
    
    db.run(
        'UPDATE users SET avatar_path = NULL WHERE id = ?',
        [req.user.id],
        function(err) {
            if (err) {
                console.error('Avatar delete DB error:', err);
                req.session.messages = [{type: 'error', text: 'Failed to remove avatar'}];
                return res.redirect('/profile');
            }
            
            req.session.messages = [{type: 'success', text: 'Avatar removed successfully'}];
            res.redirect('/profile');
        }
    );
});

module.exports = router;
