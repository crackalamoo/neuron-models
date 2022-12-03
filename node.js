//Load HTTP module
const http = require("http"),
    //fs = require("fs"),
    express = require("express"),
    path = require("path"),
    mnist = require("mnist");
//const hostname = "127.0.0.1";
const port = process.env.PORT || "3000";

var app = express();

app.engine('html', require('ejs').renderFile);
app.get('/', function(req, res) {
    //res.render("index.html", {mnist:mnist});
    res.render(path.join(__dirname, "/index.html"), {title : "Hey", mnist : mnist});
})

app.listen(port);

/*fs.readFile("./index.html", function(err, html) {
    if (err) throw err;
    http.createServer(function(request, response) {
        response.writeHeader(200, {"Content-Type": "text/html"});
        response.write(html);
        response.end();
    }).listen(port, hostname);
})*/