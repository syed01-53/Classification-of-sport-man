Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/classify_image",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Drop files here or click to upload",
        autoProcessQueue: false,
        init: function() {
            this.on("addedfile", function() {
                if (dz.files[1] != null) {
                    dz.removeFile(dz.files[0]);
                }
            });

            this.on("complete", function(file) {
                let formData = new FormData();
                formData.append("file", file);

                fetch("/classify_image", {
                    method: "POST",
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    console.log(data);
                    if (!data || data.length === 0) {
                        $("#resultHolder").hide();
                        $("#divClassTable").hide();
                        $("#error").show();
                        return;
                    }

                    let match = null;
                    let bestScore = -1;
                    for (let i = 0; i < data.length; ++i) {
                        let maxScoreForThisClass = Math.max(...data[i].class_probability);
                        if (maxScoreForThisClass > bestScore) {
                            match = data[i];
                            bestScore = maxScoreForThisClass;
                        }
                    }
                    if (match) {
                        $("#error").hide();
                        $("#resultHolder").show();
                        $("#divClassTable").show();
                        $("#resultHolder").html($(`[data-player="${match.class}"]`).html());
                        let classDictionary = match.class_dictionary;
                        for (let personName in classDictionary) {
                            let index = classDictionary[personName];
                            let probabilityScore = match.class_probability[index];
                            let elementName = "#score_" + personName;
                            $(elementName).html(probabilityScore);
                        }
                    }
                    dz.removeFile(file);
                })
                .catch(error => {
                    console.error('Error:', error);
                    $("#error").show();
                });
            });
        }
    });

    $("#submitBtn").on('click', function(e) {
        dz.processQueue();
    });
}

$(document).ready(function() {
    console.log("ready!");
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();

    init();
});
