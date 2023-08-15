console.log('custom.js loaded');

// Get all images in the document
var images = document.querySelectorAll('img');


const tooltip = document.createElement('div');
tooltip.textContent = 'hover to zoom/pan';
tooltip.style.display = 'none';
document.body.appendChild(tooltip);

images.forEach((image) => {
  image.addEventListener('mouseover', () => {
    tooltip.style.display = 'block';
  });
  image.addEventListener('mouseout', () => {
    tooltip.style.display = 'none';
  });
});


function calculateInitialScale(image) {
    // var margin = 0; // Set the desired margin in pixels
    // var W = window.innerWidth - margin * 2;
    // var H = window.innerHeight - margin * 2;
    // var scaleX = W / image.width;
    // var scaleY = H / image.height;
    // console.log('scale',W,H, image.width, image.height, image.naturalWidth,image.naturalHeight,scaleX, scaleY);
    // return Math.min(scaleX, scaleY);
    return 1; // 1 properly maximizes...
}


// Loop through the images
for (var i = 0; i < images.length; i++) {
    if (!images[i].classList.contains('sidebar-logo')) {

        // Add an event listener to each image
        images[i].addEventListener('click', function(event) {
            console.log('Image clicked');

            // Check if the image is wrapped in a link
            if (event.target.closest('a')) {
                // Prevent the default behavior of following the link
                event.preventDefault();
            }

            // Create a new image element
            var image = document.createElement('img');
            image.src = event.target.src;
            image.style.maxWidth = '100%';
            image.style.maxHeight = '100%';
            image.style.position = 'relative';

            // Calculate the initial scale of the image based on its dimensions and the dimensions of the viewport
            var initialScale =  calculateInitialScale(image);
            
            // Set the initial scale of the image
            image.style.transform = 'scale(' + initialScale + ')';
        
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
            lightbox.style.backdropFilter = 'blur(3px)';
            lightbox.style.webkitBackdropFilter = 'blur(3px)';


            // Add an event listener for double click to toggle off the light box 
            lightbox.addEventListener("dblclick", function(event){
                console.log("Light box toggled off");
                document.body.removeChild(lightbox);
            });

            // Add an event listener for escape key to toggle off the light box 
            lightbox.addEventListener("keydown", function(event){
                if(event.key === "Escape"){
                    console.log("Light box toggled off");
                    document.body.removeChild(lightbox);
                }
            });

            // Add an event listener for h key to recenter the image in the light box 
            lightbox.addEventListener("keydown", function(event){
                if(event.key === "h"){
                    console.log("Image recentered");
                    image.style.left='50%'; 
                    image.style.top='50%'; 
                    image.style.transformOrigin='50% 50%'; 
                    image.style.transform='scale('+initialScale+')';
                }
            });

            // Add an event listener for focusout to remove keydown event listeners from light box 
            lightbox.addEventListener("focusout", function(event){
                console.log("Focus out");
                lightbox.removeEventListener("keydown", arguments.callee);
            });

            // Set focus on light box to enable keydown events
            lightbox.setAttribute('tabindex', 0);
            lightbox.focus();


            // Add an event listener for the mouse wheel event
            lightbox.addEventListener('wheel', function(event) {
                // Prevent the default behavior of scrolling the page
                event.preventDefault();

                // Get the current scale of the image
                var currentScale = parseFloat(image.style.transform.replace(/.*scale\((.*)\).*/, '$1'));
                
                console.log('Scale1',initialScale);

                // Check if currentScale is NaN
                if (isNaN(currentScale)) {
                    currentScale = initialScale
                }

                // Calculate the new scale based on the wheel delta
                var newScale = currentScale + event.deltaY * -0.01;

                // Check if newScale is less than 1
                if (newScale < initialScale) {
                    newScale = initialScale;
                }

                var rect = image.getBoundingClientRect();

                // Calculate the center coordinates of the image relative to the viewport
                var centerX = rect.left + rect.width / 2;
                var centerY = rect.top + rect.height / 2;
                
                // Get the current position of the image
                var styleTransformsArray = window.getComputedStyle(image).transform.split(",");
                var currentX = parseFloat(styleTransformsArray[4]) || 0;
                var currentY = parseFloat(styleTransformsArray[5]) || 0;

                // Get the position of the mouse cursor relative to the image
                var mouseX = event.clientX - centerX;
                var mouseY = event.clientY - centerY;

                // Calculate the new position of the image based on its current position, scale, and the position of the mouse cursor
                var newX = currentX + mouseX * (1 - newScale / currentScale);
                var newY = currentY + mouseY * (1 - newScale / currentScale);

                // Set the new scale and position of the image
                image.style.transform = 'translate(' + newX + 'px, ' + newY + 'px) scale(' + newScale + ')';
            });


            // Add an event listener for the mousedown event on the image element
            image.addEventListener('mousedown', function(event) {
                // Prevent the default behavior of selecting text
                event.preventDefault();

                  var initialX = event.clientX;
                  var initialY = event.clientY;

                  var styleTransformsArray =
                    window.getComputedStyle(image).transform.split(",");
                  var currentX =
                    parseFloat(styleTransformsArray[4]) || 0;
                  var currentY =
                    parseFloat(styleTransformsArray[5]) || 0;

                  document.addEventListener("mousemove", onMouseMove);
                  document.addEventListener("mouseup", onMouseUp);

                  function onMouseMove(event) {
                    // Calculate the change in position of the mouse cursor
                    var deltaX = event.clientX - initialX;
                    var deltaY = event.clientY - initialY;
                
                    // Get the current scale of the image
                    var currentScale = parseFloat(image.style.transform.replace(/.*scale\((.*)\).*/, '$1'));
                
                    // Check if currentScale is NaN
                    if (isNaN(currentScale)) {
                        // Set currentScale to 1
                        currentScale = 1;
                    }
                
                    // Update the position and scale of the image
                    image.style.transform =
                        "translate(" +
                        (currentX + deltaX) +
                        "px," +
                        (currentY + deltaY) +
                        "px) scale(" +
                        currentScale +
                        ")";
                }
                

                function onMouseUp() {
                    document.removeEventListener("mousemove", onMouseMove);
                    document.removeEventListener("mouseup", onMouseUp);
                  }
                
              });

              lightbox.addEventListener("click", function (event) {
              if (event.target === lightbox) {
                  document.body.removeChild(lightbox);
              }



          });

          lightbox.appendChild(image);
          document.body.appendChild(lightbox);
        });
    }
}