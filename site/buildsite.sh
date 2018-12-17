cd ../ && \
python setup.py install && \
cd - && \
rm -rf apidocs modules && \
sphinx-apidoc -e \
              -o apidocs \
              ../neupy && \
tinker --build && \
python search-index/build.py --for-deploy && \
export IS_SUCCESS=1

if [ $IS_SUCCESS ]; then
	terminal-notifier -message "Site build is completed" -title "Site build"
else
	terminal-notifier -message "Site build was failed or inturapted" -title "Site build"
fi
