var express = require('express');
var ensureLogIn = require('connect-ensure-login').ensureLoggedIn;
var db = require('../db.cjs');

var router = express.Router();

function fetch_models(req, res, next) {
  db.all('SELECT * FROM models', function(err, rows) {
    if (err) { return next(err); }
    
    var models = rows.map(function(row) {
      return {
        id: row.id,
        owner_id: row.owner_id,
        title: row.title,
        date: row.date,
        stars: row.stars,
        thumb_image_path: row.thumb_image_path,
        path: row.path
      }
    });
    
    // Fixed page size
    const pageSize = 9; // Your preset number
    const page = parseInt(req.query.page) || 1;
    const totalModels = models.length;
    const totalPages = Math.ceil(totalModels / pageSize);
    const currentPage = Math.max(1, Math.min(page, totalPages || 1));
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = Math.min(startIndex + pageSize, totalModels);
    
    // Get paginated models
    const paginatedModels = models.slice(startIndex, endIndex);
    
    // Store in res.locals
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