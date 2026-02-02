const crypto = require('crypto');

const gravatar = {
    getGravatarUrl: function(email, size = 150, defaultType = 'identicon') {
        if (!email) return null;
        
        // Trim and lowercase the email
        const trimmedEmail = email.trim().toLowerCase();
        
        // Create MD5 hash
        const hash = crypto.createHash('md5').update(trimmedEmail).digest('hex');
        
        // Construct Gravatar URL
        const defaultOptions = {
            'identicon': 'identicon',
            'monsterid': 'monsterid',
            'wavatar': 'wavatar',
            'retro': 'retro',
            'robohash': 'robohash',
            'blank': 'blank',
            'mm': 'mm'
        };
        
        const defaultParam = defaultOptions[defaultType] || defaultOptions['identicon'];
        
        return `https://www.gravatar.com/avatar/${hash}?s=${size}&d=${defaultParam}&r=g`;
    }
};

module.exports = gravatar;