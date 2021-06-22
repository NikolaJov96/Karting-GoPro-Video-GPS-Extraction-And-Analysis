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

var args = process.argv.slice(2);

const gpxParam = '--gpx';
const createGpx = args.includes(gpxParam);
const jsonParam = '--json';
const createJson = args.includes(jsonParam);
const geojsonParam = '--geojson';
const createGeojson = args.includes(geojsonParam);

const files = args.filter(arg => arg != gpxParam && arg != jsonParam && arg != geojsonParam);

// Check command line arguments validity
if (!(createGpx || createJson || createGeojson) || files.length < 2) {
    console.log(`Usage: ${require('path').basename(__filename)} [${gpxParam}] [${jsonParam}] [${geojsonParam}] <video_file_name> <video_file_name>...`);
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
       
        if (createGpx)
        {
            goproTelemetry(extracted, { preset: "gpx" }).then(telemetry => {
                fs.writeFileSync("extracted_path.gpx", telemetry);
                console.log('Telemetry saved as gpx');
            });
        }
        
        if (createJson)
        {
            goproTelemetry(extracted, {}).then(telemetry => {
                fs.writeFileSync("extracted_path.json", JSON.stringify(telemetry, null, 2));
                console.log('Telemetry saved as json');
            });
        }
        
        if (createGeojson)
        {
            goproTelemetry(extracted, { preset: "geojson" }).then(telemetry => {
                fs.writeFileSync("extracted_path.geojson", JSON.stringify(telemetry, null, 2));
                console.log('Telemetry saved as geojson');
            });
        }
        
    });
    
} catch (err) {	
    console.error(err);
}
