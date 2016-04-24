gulp = require 'gulp'
coffeeify = require 'gulp-coffeeify'

gulp.task 'build', ->
  gulp.src('./src/index.coffee')
    .pipe(coffeeify())
    .pipe(gulp.dest('./build/'))

gulp.task 'default', ['build']
