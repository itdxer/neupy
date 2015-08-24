cd ../ && \
python setup.py install && \
cd - && \
rm -rf apidocs && \
sphinx-apidoc -e \
              -o apidocs \
              ../neuralpy && \
tinker --build && \
export IS_SUCCESS=1

if [ $IS_SUCCESS ]; then
	terminal-notifier -message "Site build completed" -title "Site build"
else
	terminal-notifier -message "Site build failed or inturapted" -title "Site build"
fi
