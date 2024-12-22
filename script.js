<!-- Scripts -->
<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.bundle.min.js"></script>

function overrideLightbox(event) {
    event.preventDefault(); // Prevent the lightbox effect
    goBackToMain(); // Call the function to navigate back to the main page
}

function goBackToMain() {
    window.location.href = "#art"; // Adjust to match the ID of your main page section
}