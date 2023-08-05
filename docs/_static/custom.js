// Select the "show code cell source" button
var button = document.querySelector('.code-cell .celltoolbar .btn.code-cell-toggle');

// Select the first comment in the code cell
var comment = document.querySelector('.code-cell .input_area .cm-comment');

// Change the text of the button to the text of the first comment
if (comment) {
    button.textContent = comment.textContent;
}


window.addEventListener('hashchange', function() {
    // Get the header element
    var header = document.querySelector('.wy-side-nav-search');

    // Set the background color
    header.style.backgroundColor = '#0b750a';

    // Set a delay and then change the color back
    setTimeout(function() {
        header.style.backgroundColor = '';
    }, 3000);
});





window.addEventListener('load', function() {
    // Get all images in the document
    var images = document.querySelectorAll('img');

    // Loop through the images
    for (var i = 0; i < images.length; i++) {
        // Add an event listener to each image
        images[i].addEventListener('click', function(event) {
            // Create a new image element
            var image = document.createElement('img');
            image.src = event.target.src;
            image.style.maxWidth = '100%';
            image.style.maxHeight = '100%';

            // // Create a new div element for the lightbox
            // var lightbox = document.createElement('div');
            // lightbox.style.position = 'fixed';
            // lightbox.style.top = '0';
            // lightbox.style.left = '0';
            // lightbox.style.width = '100%';
            // lightbox.style.height = '100%';
            // lightbox.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
            // lightbox.style.display = 'flex';
            // lightbox.style.justifyContent = 'center';
            // lightbox.style.alignItems = 'center';

            // Create a new div element for the lightbox
            var lightbox = document.createElement('div');
            lightbox.style.position = 'fixed';
            lightbox.style.top = '0';
            lightbox.style.left = '0';
            lightbox.style.width = '100%';
            lightbox.style.height = '100%';
            lightbox.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
            lightbox.style.display = 'flex';
            lightbox.style.justifyContent = 'center';
            lightbox.style.alignItems = 'center';
            lightbox.style.zIndex = '9999';
            lightbox.style.backdropFilter = 'blur(5px)'; // Add this line


            // Add the image to the lightbox
            lightbox.appendChild(image);

            // Add the lightbox to the body
            document.body.appendChild(lightbox);

            // Add an event listener to the lightbox
            lightbox.addEventListener('click', function() {
                // Remove the lightbox from the body
                document.body.removeChild(lightbox);
            });
        });
    }
});
