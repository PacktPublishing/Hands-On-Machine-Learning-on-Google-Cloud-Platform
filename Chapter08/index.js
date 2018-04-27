const functions = require('firebase-functions');
const Translate = require('@google-cloud/translate');
const admin = require("firebase-admin")

//setting connection to db
admin.initializeApp();


const translate = new Translate({
    projectId: 'mytranslator-c656d'
});
// to enable calling by users
exports.translateMessage=functions.https.onRequest((req,res) => 
{
const input = req.query.text;

translate.translate(input,'en').then(results => 
{
    const output = results[0];
    console.log(output);

    const db = admin.database();

    var ref = db.ref("/translateMessageStats");
   

// update database
var dataRef= ref.child(input);

dataRef.once('value', function(snapshot) {
    if (snapshot.exists()) {
        dataRef.update({"count":snapshot.val().count+1});
        console.log("data exists")
    }
    else
    {
        console.log("data does not exist")
        dataRef.update({"count":1});

    }
  });

    return res.send(JSON.stringify(output));
})


});