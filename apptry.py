from flask import Flask, redirect, url_for, request 
from WMD import WordS_Mover_Distance
app = Flask(__name__) 
  
@app.route('/success/<name>') 
def success(name): 
   return 'welcome %s' % name 
  
#s2 = "On first day as president, Biden to issue 17 executive actions on COVID, climate change, immigration and more"

@app.route('/login',methods = ['POST', 'GET']) 
def login(): 
   global wmd_Obj
   if request.method == 'POST': 
      s1 = request.form['s1']
      s2 = request.form['s2']
      dist = wmd_Obj.TextSimilarity(s1, s2) 
      return redirect(url_for('success',name = dist)) 
   else: 
      user = request.args.get('nm') 
      return redirect(url_for('success',name = user)) 

if __name__ == '__main__': 
   wmd_Obj = WordS_Mover_Distance()
   app.run(debug = True)