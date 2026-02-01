var express = require('express');
var ensureLogIn = require('connect-ensure-login').ensureLoggedIn;
var db = require('../db.cjs');

var router = express.Router();

function delete_model_from_db(model_id) {
    db.run('DELETE FROM models WHERE id = ?', model_id,
    function(err) {
        if (err) {
            return console.error(err.message);
        }
    });
}

function fetch_models(req, res, next) {
    console.log("req.query.page: ", req.query.page);
    const page = req.query.page ? parseInt(req.query.page, 10) : 1;
    const itemsPerPage = 8; // Changed from 200 to 8 for pagination
    const offset = (page - 1) * itemsPerPage;

    let sql = 'SELECT COUNT(*) AS count FROM models';

    db.get(sql, [], (err, row) => {
        if (err) {
            throw err;
        }

        const totalCount = row.count;
        const totalPages = Math.ceil(totalCount / itemsPerPage);

        db.all('SELECT * FROM models ORDER BY id LIMIT ? OFFSET ?', [itemsPerPage, offset], (err, rows) => {
            if (err) {
                res.status(500).send(err.message);
                return;
            }
            
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
            
            res.locals.models = models;
            res.locals.row_count = totalCount;
            res.locals.page = page;
            res.locals.totalPages = totalPages;
            res.locals.itemsPerPage = itemsPerPage;
            
            console.log(`Page: ${page}, Total: ${totalCount}, Pages: ${totalPages}`);
            next();
        });
    });
}

router.get('/delete', fetch_models, (req, res) => {
    res.render('delete', { 
        user: req.user,
        models: res.locals.models || [],
        row_count: res.locals.row_count || 0,
        page: res.locals.page || 1,
        totalPages: res.locals.totalPages || 1,
        itemsPerPage: res.locals.itemsPerPage || 8
    });
});

// Define a route for model deletion.
router.post('/delete', (req, res) => {
    const row_data = req.body.row_data;
    console.log("row_data: ", row_data);
    const model_id = row_data[0];
    console.log("model_id: ", model_id);
    
    delete_model_from_db(model_id);
    
    res.json({ 
        message: 'Model deleted successfully!',
        success: true 
    });
});

module.exports = router;