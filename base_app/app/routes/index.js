var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express' });
});


router.get('/neuralnetwork', function(req, res, next) {
  res.render('neuralnetwork', {
    title: 'neuralnetwork'
  });
});

module.exports = router;
