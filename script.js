function overrideLightbox(event) {
    event.preventDefault(); // Prevent the lightbox effect
    goBackToMain(); // Call the function to navigate back to the main page
}

function goBackToMain() {
    window.location.href = "#art"; // Adjust to match the ID of your main page section
}
