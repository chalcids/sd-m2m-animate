function submit_m2m_animate() {
    rememberGallerySelection('m2m_animate_gallery')
    showSubmitButtons('m2manimate', false)
    showResultVideo('m2m_animate', false)

    var id = randomId()
    requestProgress(id, gradioApp().getElementById('m2m_animate_gallery_container'), gradioApp().getElementById('m2m_animate_gallery'), function () {
        showSubmitButtons('m2manimate', true)
        showResultVideo('m2m_animate', true)
    })

    var res = create_submit_args(arguments)
    res[0] = id
    return res
}

function showResultVideo(tabname, show) {
    gradioApp().getElementById(tabname + '_video').style.display = show ? "block" : "none"
    gradioApp().getElementById(tabname + '_gallery').style.display = show ? "none" : "block"

}


function showModnetModels() {
    var check = arguments[0]
    gradioApp().getElementById('m2m_animate_modnet_model').style.display = check ? "block" : "none"
    gradioApp().getElementById('m2m_animate_merge_background').style.display = check ? "block" : "none"
    return []
}

function switchModnetMode() {
    let mode = arguments[0]

    if (mode === 'Clear' || mode === 'Origin' || mode === 'Green' || mode === 'Image') {
        gradioApp().getElementById('modnet_background_movie').style.display = "none"
        gradioApp().getElementById('modnet_background_image').style.display = "block"
    } else {
        gradioApp().getElementById('modnet_background_movie').style.display = "block"
        gradioApp().getElementById('modnet_background_image').style.display = "none"
    }

    return []
}


function copy_from(type) {
    return []
}

function submitTestFrame() {
    showSubmitButtons('m2mtestframe', false);
    showResultVideo('m2m_animate', false)
    var id = randomId();
    localSet("m2m_animate_test_task_id", id);

    requestProgress(id, gradioApp().getElementById('m2m_animate_gallery_container'), gradioApp().getElementById('m2m_animate_gallery'), function() {
        showSubmitButtons('m2mtestframe', true);
        localRemove("m2m_animate_test_task_id");
    });

    var res = create_submit_args(arguments);

    res[0] = id;

    return res;
}

function setRandomSeed(elem_id) {
    var input = gradioApp().querySelector("#" + elem_id + " input");
    if (!input) return [];

    input.value = "-1";
    updateInput(input);
    return [];
}