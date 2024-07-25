function toggleMenu() {
    const menu = document.querySelector(".menu-links");
    const icon = document.querySelector(".hamburger-icon");
    menu.classList.toggle("open");
    icon.classList.toggle("open");
  }
  
  // Show or hide the button based on scroll position
  window.onscroll = function() {scrollFunction()};
  
  function scrollFunction() {
    const scrollTopBtn = document.getElementById("scrollTopBtn");
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
      scrollTopBtn.style.display = "block";
    } else {
      scrollTopBtn.style.display = "none";
    }
  }
  
  // Scroll to the top when the button is clicked
  document.getElementById("scrollTopBtn").onclick = function() {
    document.body.scrollTop = 0;
    document.documentElement.scrollTop = 0;
  };

  document.addEventListener('DOMContentLoaded', function() {
    console.log('JavaScript is loaded and working!');

    // Redirect to the basic screen page when the assess button is clicked
    document.getElementById('assess').addEventListener('click', function() {
        window.location.href = "/basic-screen";
    });
});

function toggleMenu() {
    const menu = document.querySelector(".menu-links");
    const icon = document.querySelector(".hamburger-icon");
    menu.classList.toggle("open");
    icon.classList.toggle("open");
}

// Show or hide the button based on scroll position
window.onscroll = function() {scrollFunction()};

function scrollFunction() {
    const scrollTopBtn = document.getElementById("scrollTopBtn");
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
        scrollTopBtn.style.display = "block";
    } else {
        scrollTopBtn.style.display = "none";
    }
}

// Scroll to the top when the button is clicked
document.getElementById("scrollTopBtn").onclick = function() {
    document.body.scrollTop = 0;
    document.documentElement.scrollTop = 0;
};
  
  
  