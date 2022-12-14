const express = require("express"),
    path = require("path"),
    mnist = require("mnist");

const port = process.env.PORT || "3000";
const html_dir = "/index_node.html";
const js_dir = "public";

var app = express();

app.engine('html', require('ejs').renderFile);
app.use(express.static(path.join(__dirname, js_dir)));
app.get('/', function(req, res) {
    res.render(path.join(__dirname, html_dir), {mnist : mnist});
});

app.listen(port);
