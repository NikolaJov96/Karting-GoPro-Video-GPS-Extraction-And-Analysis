const gpmfExtract = require('gpmf-extract');
const goproTelemetry = require(`gopro-telemetry`);
const fs = require('fs');

/**
 * Function for loading big files, see readme for more info
 *
 * @param {string} path Path to the file to be loaded
 */
function bufferAppender(path) {
    return function (mp4boxFile) {
        var stream = fs.createReadStream(path, { highWaterMark: 4 * 1024 * 1024 });
        var bytesRead = 0;
        stream.on('end', () => {
            mp4boxFile.flush();
        });
        stream.on('data', chunk => {
            var arrayBuffer = new Uint8Array(chunk).buffer;
            arrayBuffer.fileStart = bytesRead;
            mp4boxFile.appendBuffer(arrayBuffer);
            bytesRead += chunk.length;
        });
        stream.resume();
    };
}

// Parse command line arguments

var argv = require('minimist')(process.argv.slice(2));

const gpxOutFile = argv.gpx;
const jsonOutFile = argv.json;
const geojsonOutFile = argv.geojson;

const files = argv._;

// Check command line arguments validity
if ((!gpxOutFile && !jsonOutFile && !geojsonOutFile) || files.length < 1) {
    console.log(`Usage: ${require('path').basename(__filename)} [--gpx <gpx_out_file>] [--json <json_out_file>] [--geojson <geojson_out_file>] <video_file_name> <video_file_name>...`);
    process.exit(1);
}

// Generate output
try {
    // Create a promise for each input file
    const promises = []
    for (const file of files) {
        promises.push(gpmfExtract(bufferAppender(file)));
    }

    // When all files are loaded, generate the output
    Promise.all(promises).then(extracted => {

        if (gpxOutFile)
        {
            goproTelemetry(extracted, { preset: "gpx" }).then(telemetry => {
                fs.writeFileSync(gpxOutFile, telemetry);
                console.log('Telemetry saved as gpx');
            });
        }

        if (jsonOutFile)
        {
            goproTelemetry(extracted, {}).then(telemetry => {
                fs.writeFileSync(jsonOutFile, JSON.stringify(telemetry, null, 2));
                console.log('Telemetry saved as json');
            });
        }

        if (geojsonOutFile)
        {
            goproTelemetry(extracted, { preset: "geojson" }).then(telemetry => {
                fs.writeFileSync(geojsonOutFile, JSON.stringify(telemetry, null, 2));
                console.log('Telemetry saved as geojson');
            });
        }

    });

} catch (err) {
    console.error(err);
}
