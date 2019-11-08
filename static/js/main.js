const hasAnnotation = document.getElementById("hasAnnotation")

function annotationDownload() {
    if (hasAnnotation.value !== "True") {
        alert("Для завантаженя звіту будь-ласка заповніть аннотацію");
        return ;
    }
    console.log('report');
    let w = window.open('/report/', '_blank');
    w.focus();
}